# LaTeX

## How to compile

```bash
xelatex main.tex
```

## How to use VSCode LaTeX Workshop

Add the following configuration to your `settings.json`:
```json
{
    "latex-workshop.latex.tools": [{
        "name": "latexmk",
        "command": "latexmk",
        "args": [
            "-xelatex",
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ]
    }]
}