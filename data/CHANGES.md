# Knowledge base changes

`original/ramaiah_official_faq_dataset_cleaned.csv` is the file used in Colab, kept unchanged for reference.
`ruas_faqs.csv` is the file the app uses. Differences:

| FAQ | Field | Before | After | Why |
|---|---|---|---|---|
| 43 | role | admin | student | It holds the admissions website, phone and email. As `admin`, students could never be told how to contact admissions. |
| 28 | category | Admissions | Admission_Process | Every other admissions FAQ uses `Admission_Process`. One category per topic keeps category-scoped rules working. |
| all | file encoding | UTF-8 with a hidden byte-order mark | plain UTF-8 | The mark made the first column read as `﻿id` instead of `id` in some tools. (The loader now handles either.) |

Still worth reviewing with the university (not changed):

- Only one FAQ (#28, intake capacity) is admin-only, and its answer is public information. The role feature works but has nothing sensitive to protect yet.
- Six answers only say "check the official website" (#1, 10, 13, 20, 28, 31).
- Time-sensitive facts (fee amount, application and exam months) have no year attached.
