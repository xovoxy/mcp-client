import asyncio
import os
import json,sys
from typing import Optional, Dict
from contextlib import AsyncExitStack
from mcp import ClientSession,  StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # load env variables form.env

class MultiServerMCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.base_url,
        )
        self.sessions: Dict[str, ClientSession] = {}
        self.toos_by_session: Dict[str, list] = {}
        self.all_tools = []
    async def connect_to_server(self, servers: dict):
        for server_name, script_path in servers.items():
            session = await self._start_one_server(script_path)
            self.sessions[server_name]=session
            
            resp = await session.list_tools()
            self.toos_by_session[server_name]=resp.tools
            
            for tool in resp.tools:
                function_name = f"{server_name}_{tool.name}"
                self.all_tools.append({
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                })
        self.all_tools = await self.transform_json(self.all_tools)
        print("\n✅ 已连接到下列服务器:")
        for name in servers:
            print(f"    - {name}: {servers[name]}")
        print("\n汇总的工具：")

        for t in self.all_tools:
            print(f"    - {t['function']['name']}")
                

    async def transform_json(self, json2_data) -> list:
        result = []

        for item in json2_data:
            if not isinstance(item, dict) or "type" not in item or "function" not in item:
                continue
            
            old_func = item["function"]
            
            if not isinstance(old_func, dict) or "name" not in old_func or "description" not in old_func:
                continue
            
            new_func = {
                "name": old_func["name"],
                "description": old_func["description"],
                "parameters": {}
            }
            
            if "input_schema" in old_func and isinstance(old_func["input_schema"], dict):
                old_schema = old_func["input_schema"]

                new_func["parameters"]["type"] = old_schema.get("type", "object")
                new_func["parameters"]["properties"] = old_schema.get("properties", {})
                new_func["parameters"]["required"] = old_schema.get("required", [])

            new_item = {
                "type": item["type"],
                "function": new_func,
            }
            
            result.append(new_item)
        return result
            

    async def _start_one_server(self, script_path: str) -> ClientSession:
        is_python = script_path.endswith(".py")
        is_js = script_path.endswith(".js")
        if not is_python and not is_js:
            raise ValueError("Unsupported file type. Only .py and .js files are supported.")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None,
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return session
    
    
    async def chat_base(self, messages: list) -> list:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.all_tools,
        )
        
        if response.choices[0].finish_reason == "tool_calls":
            while True:
                messages = await self.create_function_response_message(messages, response)
                print(messages)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.all_tools,
                )
                print(response)
                if response.choices[0].finish_reason != "tool_calls":
                    break
        
        return response
                
    async def create_function_response_message(self, messages, response):
        function_call_messages = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message.model_dump())
        
        for function_call_message in function_call_messages:
            tool_name = function_call_message.function.name
            print(f'{function_call_message=}')
            tool_args = json.loads(function_call_message.function.arguments)
            function_response = await self._call_mcp_tool(tool_name, tool_args)
            print(f'{function_response=}')
            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                    "tool_call_id": function_call_message.id,
                }
            )
        return messages
    
    async def process_query(self, user_query: str) -> str:
        messages = [{"role": "user", "content": user_query}]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.all_tools,           
        )
        content=response.choices[0]
        
        if content.finish_reason == "tool_calls":
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"\n[ 调用工具： {tool_name}, 参数：{tool_args} ]\n")
            
            result = await self._call_mcp_tool(tool_name, tool_args)
            
            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        
        return content.message.content
            
            
            
    async def _call_mcp_tool(self, tool_full_name: str, tool_args: dict) -> str:
        
        parts = tool_full_name.split("_", 1)
        if len(parts)  != 2:
            return f"无效的工具名称：{tool_full_name}"
        
        server_name, tool_name = parts
        session = self.sessions.get(server_name)
        if not session:
            return f"找不到服务器：{server_name}"
        
        
        resp = await session.call_tool(tool_name, tool_args)
        return resp.content[0].text if resp.content else "工具执行无输出"
    
    
    async def chat_loop(self):
        print("\n 多服务器 MCP + 最新 Function Calling 客户端已启动！输入'quit'退出。 ")
        messages = []
        
        while True:
            query = input("\n你： ").strip()
            if query.lower() == "quit":
                break
            try:
                messages.append({"role": "user", "content": query})
                messages = messages[-20: ]
                response = await self.chat_base(messages)
                messages.append(response.choices[0].message.model_dump())
                result = response.choices[0].message.content
                
                print(f"\nDeepSeek: {result}")
            except Exception as e:
                print(f"\n 调用过程出错：{e}")
                
    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    servers = {
        "file": "../mcp-server-demo/main.py",
    }
    
    client = MultiServerMCPClient()
    try:
        await client.connect_to_server(servers)
        await client.chat_loop()
    finally:
        await client.cleanup()
        
if __name__=="__main__":
    asyncio.run(main())