# Server Side Settings
~/.vscode-server/data/Machine/settings.json
```JSON
{
  "terminal.integrated.gpuAcceleration": "on",
  "terminal.integrated.customGlyphs": false,
  "terminal.integrated.allowChords": false,
  "terminal.integrated.suggest.quickSuggestions": false,
  "terminal.integrated.drawBoldTextInBrightColors": false,
  "terminal.integrated.enableMultiLinePasteWarning": "never",
  "terminal.integrated.shellIntegration.decorationsEnabled": "never",
  "terminal.integrated.suggest.suggestOnTriggerCharacters": false,
  "terminal.integrated.tabs.enableAnimation": false
}
```

# Fix TableGen IntelliSense
.vscode/settings.json
```JSON
{
  "mlir.tablegen_compilation_databases": [
    "./examples/triton_encodings/build/tablegen_compile_commands.yml"
  ]
}
```