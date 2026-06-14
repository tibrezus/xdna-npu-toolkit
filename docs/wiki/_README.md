# Wiki content (bootstrap pending)

GitHub does not expose wiki creation via API — the wiki git repo only exists
after the **first page is saved through the web UI** (one-time manual click).
Once you create any page at https://github.com/tibrezus/xdna-npu-toolkit/wiki
(click "Create the first page"), the `.wiki.git` repo appears and these files
can be pushed to it wholesale:

```bash
git clone https://github.com/tibrezus/xdna-npu-toolkit.wiki.git
cp docs/wiki/*.md <wiki-clone>/   # except _README.md
cd <wiki-clone> && git add -A && git commit -m "Populate wiki" && git push
```

The same content is also in `docs/EMBEDDINGS-WALKTHROUGH.md` so it's available
in-repo regardless.
