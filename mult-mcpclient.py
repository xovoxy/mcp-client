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
                

    async def transform_json(self, json2_data):
        result = []

        for item in json2_data:
            if not isinstance(item, dict) or "type" not in item or "function" not in item:
                continue
            
            old_func = item["function"]
            
            if not isinstance(old_func, dict) or "name" not in old_func or "description" not in item:
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
            ClientSession.create(read_stream, write_stream)
        )
        await session.initialize()
        return session