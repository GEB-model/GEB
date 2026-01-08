# GEB MCP Server

The GEB MCP (Model Context Protocol) server allows you to interact with the GEB model using AI assistants like the Google Gemini CLI. This enables you to configure, build, and run models using natural language prompts.

## Automated Setup via Agent Prompt

To set up the model you can use the following prompt in the [gemini-cli](https://geminicli.com/docs/get-started/) or follow the instructions yourself.

**Prompt:**

>   **MCP Configuration**: Create a file at `.gemini/settings.json` with the following content:
>   ```json
>   {
>       "mcpServers": {
>           "geb": {
>               "command": "uv",
>               "args": ["run", "geb/mcp_server.py"]
>           }
>       }
>   }
>   ```

## Usage

Once the Gemini CLI is running and connected to the GEB server, you can use natural language to interact with the model.

### Example Prompts

**Initialize and build a model:**

> "Initialize a new GEB model for the Geul basin using the example configuration, then build it."

**Run a simulation:**

> "Run the model spinup and run."