# GEB MCP Server

The GEB MCP (Model Context Protocol) server allows you to interact with the GEB model using AI assistants like the Google Gemini CLI. This enables you to configure, build, and run models using natural language prompts.

## Automated Setup via Agent Prompt

To set up the model you can use the following prompt in the [gemini-cli](https://geminicli.com/docs/get-started/) or follow the instructions yourself.

**Prompt:**

```
To configure the MCP server, create a file at `.gemini/settings.json` with the following content:
   
{
    "mcpServers": {
        "geb": {
            "command": "uv",
            "args": ["run", "geb/mcp_server.py"]
        }
    }
}

Then remind the user to restart gemini before the mcp server can be used.
```

## Usage

If you added the configuration through gemini, you will need to restart gemini (press ctrl+c twice). Once the Gemini CLI is running again, you can use natural language to interact with GEB.

### Example Prompts

**Initialize and build a model:**

> "Initialize a new GEB model for the Geul basin using the example configuration, then build it."

**Run a simulation:**

> "Run the model spinup and run."