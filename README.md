This is the section for data ingestion and preprocessing for the Allegorical Plato project.

The data for this project is sourced entirely from the PerseusDL/canonical-greekLit repository. Deep thanks to them for
making this data available.

Run `uv run scraping` to rebuild the lossless corpus data.

Ancient Greek linguistic enrichment is deliberately separate from TEI parsing:

```sh
uv run --extra nlp scraping-models
uv run --extra nlp scraping-enrich
```

The first command explicitly downloads Stanza's Ancient Greek Perseus model. The
second preserves `text_raw` and `text_clean`, adds a materialized `text_topic`
lemma stream, and writes `tokens.parquet` plus `linguistic-run.json` under
`data/corpus`. Every excluded particle, dialogue formula, function word, and
proper name remains available in the token table with an `exclusion_reason`.
