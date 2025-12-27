# evalyn init

Initialize a configuration file with default settings.

## Usage

```bash
evalyn init [OPTIONS]
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output FILE` | evalyn.yaml | Output file path |
| `--force` | false | Overwrite existing file |

## Examples

### Create default config
```bash
evalyn init
```

### Custom location
```bash
evalyn init --output ~/.evalyn.yaml
```

### Overwrite existing
```bash
evalyn init --force
```

## Generated Config

```yaml
# Evalyn Configuration
# API Keys (use env vars or set directly)
api_keys:
  gemini: "${GEMINI_API_KEY}"
  openai: "${OPENAI_API_KEY}"

# Default model for LLM operations
model: "gemini-2.5-flash-lite"

# Default project settings
defaults:
  project: null
  version: null
```

## Environment Variable Expansion

The config supports `${VAR_NAME}` syntax for environment variables:

```yaml
api_keys:
  gemini: "${GEMINI_API_KEY}"  # Reads from environment
  openai: "sk-..."             # Or set directly (not recommended)
```

## Sample Output

```
Created evalyn.yaml

Set your API key:
  export GEMINI_API_KEY='your-key'

  # or edit evalyn.yaml directly
```

## See Also

- [one-click](one-click.md) - Uses config for defaults
- [suggest-metrics](suggest-metrics.md) - Uses config for model settings
