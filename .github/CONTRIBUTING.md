# Contributing Guidelines

This document defines the required conventions for branching, commits, pull requests, and code style across this repository. Consistency and clarity are mandatory for all contributions.

---

## 1. Branching Strategy

- Use long-lived research and feature branches.
- Each branch isolates a subsystem, experiment, or focused enhancement.

### Workflow Rules

1. Never commit directly to `main`.
2. Always create a new branch before starting work:

   ```bash
   git checkout main
   git pull origin main
   git checkout -b exp/kernel-memory
   ```
3. Keep branches focused. One topic per branch.
4. Open a Pull Request when ready for review.

## 2. Commit Message Convention

- Use the following structure:
```
<type>: <short, imperative description>

[optional body]

[optional footer]
```
- Subject line must be under 72 characters.
* Use **imperative mood** (“Add,” “Fix,” “Update”).

### Allowed Types

| Type           | Description                                                       |
| :------------- | :---------------------------------------------------------------- |
| **`feat`**     | Add or modify functionality.                                      |
| **`fix`**      | Correct a bug or issue.                                           |
| **`docs`**     | Documentation or comments only.                                   |
| **`style`**    | Code formatting or readability changes.                           |
| **`refactor`** | Code structure changes without altering behavior.                 |
| **`perf`**     | Performance or efficiency improvements.                           |
| **`test`**     | Add or modify tests.                                              |
| **`chore`**    | General maintenance or configuration updates.                     |
| **`research`** | Experimental commits related to testing hypotheses or benchmarks. |
| **`exp`**      | Prototype-level or exploratory commits.                           |

## 3. Pull Requests and Review

- Open a Pull Request targeting `main`.
- All PRs must be reviewed by at least one other member.
- Use Squash and Merge to maintain clean history.

## 4. Coding Standards

- **C:** snake_case for functions/variables, PascalCase for structs, Doxygen-style comments.
- **Python:** PEP 8, 4-space indentation, snake_case for functions/variables, PascalCase for classes, use `black` for formatting.


## 5. Best Practices

| Area              | Do                                                        | Don’t                              |
| :---------------- | :-------------------------------------------------------- | :--------------------------------- |
| **Branching**     | Use clear, descriptive names`. | Push directly to `main`.           |
| **Committing**    | Keep each commit focused and atomic.                      | Mix unrelated changes.             |
| **Messages**      | Follow the commit format exactly.                         | Use vague or incomplete summaries. |
| **Pull Requests** | Test and review before merging.                           | Merge untested or unreviewed code. |
| **Cleanup**       | Remove experimental leftovers before merge.               | Leave temporary or debug files.    |
| **Code**          | Format, document, and lint before commit.                 | Skip docstrings or comments. 
