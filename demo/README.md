# Live case study

`index.html` is the page served at [credit-risk-alven.vercel.app](https://credit-risk-alven.vercel.app). It is
a single static file: the figures are embedded as JSON in a `<script type="application/json">` block and the
page renders the approval-policy explorer, the decile chart, the calibration table, the reason-code example
and the points table from that block. Nothing on the page is retyped.

The embedded JSON is built from the executed outputs in this repository:

| Page element | Source |
|---|---|
| Headline metrics, deciles, calibration, example decline, fairness checks | `outputs/results.json` |
| Approval-policy explorer (50% to 100% in 1% steps) | `outputs/val_scores.parquet`, 45% LGD, credit amount as exposure; the 90/80/70% rows reproduce `outputs/business_impact.json` to the digit |
| Points table (top 8 characteristics by IV) | `outputs/scorecard_points.json` |
| LightGBM benchmark | `outputs/challengers.json` |

To rebuild the page after the pipeline is re-run, regenerate the JSON from those files and replace the
`__DATA__` block, then redeploy from a clean folder:

```bash
mkdir site && cp demo/index.html site/ && cd site
npx vercel link --project credit-risk-alven --scope alven1 --yes && npx vercel deploy --prod --scope alven1
```

This page replaced, on 2026-10-17, an earlier demo that showed figures from the previous pipeline.
That demo's data file and its four charts were removed the same day.
