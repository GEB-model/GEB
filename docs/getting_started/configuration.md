# Configuration

The configuration of the GEB model is defined using a YAML file. The structure of the configuration is validated using Pydantic models. Below is the reference for the configuration options.

::: geb.config_schema.Config
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      heading_level: 3

## Detailed Configuration Options

::: geb.config_schema
    options:
      show_root_heading: false
      show_source: false
      members_order: source
      filters: ["!^Config$"]
      heading_level: 3
