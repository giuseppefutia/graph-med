# VSCode Google Colab Plugin — Troubleshooting

## Common Issue: Kernel Fails to Start / Run All Blocked

**Error example:**
```
Failed to start the Kernel 'Python 3 (ipykernel)'.
Unable to get resolved server information for google.colab:colab:<session-id>
```

**Cause:** The Colab session expired or disconnected, and VSCode is still referencing a stale session ID.

## Fix Steps (in order)

### 1. Reconnect to a Fresh Colab Kernel
- Click the **kernel name** in the top-right of the notebook
- Select **"Select Another Kernel"** → **"Connect to Google Colab"**
- Re-authenticate if prompted
- Try **Run All** again

### 2. Restart the Kernel
- `Cmd+Shift+P` → **"Jupyter: Restart Kernel"**

### 3. Clear Jupyter Servers
- `Cmd+Shift+P` → **"Jupyter: Clear All Jupyter Servers"**
- Reopen the notebook and reconnect to Colab

### 4. Reload VSCode Window
- `Cmd+Shift+P` → **"Developer: Reload Window"**

### 5. Clear Stale Kernel Files
```bash
rm -rf ~/.local/share/jupyter/runtime/kernel-*.json
```
Then restart VSCode and reconnect.

### 6. Re-specify Jupyter Server
- `Cmd+Shift+P` → **"Jupyter: Specify Jupyter Server for Connections"**
- Select the Colab option to re-authenticate

## Fallback: Use the Colab Web UI

If the plugin remains unresponsive, run notebooks directly in the browser:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload the `.ipynb` file (or open from Google Drive)
3. Run All from the browser

The browser UI is more stable for long training runs.

## Prerequisites

- **Extension required:** `googlecolab.vscode-google-colab` must be installed and enabled in VSCode
