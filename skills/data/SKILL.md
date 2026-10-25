---
name: data
description: "Analyze local datasets with shell and Python: inspect quality, script results, state uncertainty."
aliases: data-analysis, datasets, analytics
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Data

Data analysis answers a question from observed records. The output is only as
trustworthy as the provenance, schema, and cleaning behind it. Explore freely,
but make every reported number reproducible and separate observation from
interpretation.

## Use this skill when

- A CSV, JSON, log, Parquet, SQLite, or local data file needs analysis.
- The user asks for counts, trends, segments, anomalies, or a data quality check.
- A report contains numbers that need to be traced back to a source and method.
- You need a small reproducible script rather than manual spreadsheet edits.

## Do not use this skill when

- The data must be fetched from a remote system you cannot access. Name the gap
  and ask for an export or query result.
- The task is causal inference from observational data without identification
  evidence. Report association and its limits instead.
- The request is a production pipeline change with backfills or downstream
  readers; treat that as an engineering change with its own review.

## Workflow

### 1. Scope the question and the data

Write down:

- the decision the analysis supports;
- the source path, owner, and extraction time;
- the row/record scope: population, filters, date range, and exclusions;
- the schema and join keys;
- what would make the result invalid.

Ask for the smallest missing artifact if the file or schema is unavailable. Do
not invent a dataset, a row count, or a trend.

### 2. Inspect before analyzing

Start with cheap, read-only checks:

```sh
ls -lh data/
file data/*.csv
wc -l data/*.csv
head -n 5 data/example.csv
python -c "import csv; r=csv.reader(open('data/example.csv')); print(next(r))"
```

Use Python with the standard library when possible; use pandas/polars only if
already available. Check:

- delimiter, quoting, encoding, header row, and extra columns;
- inferred types and values that do not parse;
- missing values, empty strings, sentinel values, and mixed units;
- duplicate records and duplicate primary keys;
- ranges, negative values, future dates, time zones, and daylight-saving edges;
- category spellings and trailing whitespace;
- join cardinality and unmatched keys before joining.

### 3. Establish data quality explicitly

Produce a small table of row counts at each stage:

```
raw rows            120,431
after header fix    120,430
after dedupe        119,882
after date filter   118,005
joined to accounts  117,640 (365 unmatched)
```

For each exclusion, state why. Quantify missingness per column; do not silently
drop rows. Keep the raw file read-only and write derived outputs elsewhere. If a
cleaning decision could change the answer, run the analysis with and without it
and report the sensitivity.

### 4. Write a reproducible script

Prefer a checked-in script over a series of ad hoc commands once the analysis
matters. The script should:

- accept input and output paths as arguments or constants near the top;
- read raw data without modifying it;
- apply cleaning in named, testable steps;
- set random seeds and report library versions;
- write a machine-readable result plus a short summary;
- fail loudly on unexpected schema, not silently coerce;
- print the commands or a one-line reproduction instruction.

Keep exploratory one-liners for discovery. Promote anything that produces a
reported number into the script so it can be rerun.

### 5. Analyze with the right method

Choose the method from the question, not from the tool you know:

- counts and rates: state the denominator;
- trends: use comparable time buckets and note seasonality;
- segments: report group sizes, not just percentages;
- anomalies: define the rule before looking for outliers;
- correlation: report the coefficient, sample size, and confounding limits;
- joins: check multiplicity; a one-to-many join can inflate totals.

Check the result on a small manual sample. Reconcile totals against the source.
If two queries should agree, make them agree or explain why they do not.

### 6. Communicate uncertainty

Separate three layers in the output:

- **Observed** - direct counts, fields, and records in the supplied data.
- **Inferred** - cleaning assumptions, join logic, and interpretation.
- **Unknown** - missing data, unobserved populations, and causal claims that the
  design cannot support.

Use confidence intervals or ranges when estimating. Round to a precision the data
supports; "about 42%" is honest when the sample is small. Do not describe an
association as an effect, and do not extrapolate beyond the observed scope.

## Deliverable

Return or write a compact summary:

```
Question: ...
Data: <path> — <rows> rows, extracted <date>, scope <filters>
Method: <script path> — <key steps>
Findings: <number> — <denominator> — <uncertainty>
Quality: <missing/duplicate/join issues>
Limits: <what this cannot establish>
Reproduce: <exact command>
```

Charts should have labeled axes, units, a stated time range, and a source note.
Do not truncate axes to exaggerate a difference.

## Completion checklist

- [ ] Question, population, and data provenance stated.
- [ ] Schema, types, encoding, and key uniqueness checked.
- [ ] Missingness, duplicates, and exclusions quantified.
- [ ] Raw inputs untouched; derived outputs written separately.
- [ ] Reported numbers come from a rerunnable script.
- [ ] Totals reconcile or discrepancies are explained.
- [ ] Observed, inferred, and unknown claims separated.
- [ ] Uncertainty and limits stated.
- [ ] No causal claim without identification evidence.

## Failure modes

- Reading only the first rows and assuming the schema holds.
- Dropping nulls without reporting how many or why.
- A join that multiplies rows and silently inflates a total.
- Reporting a percentage without its denominator.
- Treating correlation as causation because the chart looks convincing.
- Hard-coding a result after manual spreadsheet edits.
- Mixing time zones or comparing partial periods.
- Presenting a single run as robust without a sensitivity check.
