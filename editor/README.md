# Floor Plan editor for "Promote"

Copy your `floor_plan.html` into this folder:

```
editor/floor_plan.html
```

Then restart `fpvec-demo`. The demo's **Promote → Open in Floor Plan editor** button opens it at
`http://localhost:8000/editor` with the traced (and edited) plan already loaded. The editor restores its last
drawing from the browser's storage on start, and the demo puts the plan there, on the same address.

To keep the editor somewhere else, set an environment variable instead:

```powershell
$env:FPVEC_EDITOR = "C:\path\to\floor_plan.html"
fpvec-demo
```

`floor_plan.html` is not committed here. If you want it in this repository, remove the line `editor/floor_plan.html` from `.gitignore`.
