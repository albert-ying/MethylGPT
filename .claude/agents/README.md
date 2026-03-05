# MethylGPT Refactoring Agent Team

## Team Structure

| Agent | File | Role |
|-------|------|------|
| **Team Lead** | `team-lead.md` | Orchestrates work, plans tasks, reviews outputs |
| **Coder** | `coder.md` | Implements refactoring, writes clean code |
| **QA/Evaluator** | `qa-evaluator.md` | Tests, validates GPU demos, regression checks |
| **Documenter** | `documenter.md` | Docstrings, README, tutorials, architecture docs |

## How to Use

### As Team Lead (orchestration mode)
```
Read .claude/agents/team-lead.md, then plan and delegate tasks.
```

### As individual agent
```
Read .claude/agents/coder.md and work on assigned task.
```

### Typical workflow
1. **Team Lead** reviews current state, creates task list
2. **Coder** implements changes (code refactoring, new modules)
3. **QA/Evaluator** tests changes (unit tests, GPU demos)
4. **Documenter** updates docs to match new code
5. **Team Lead** reviews all outputs, commits, reports progress

## Environment
- **Conda**: `methygpt` (`/home/kying/anaconda3/envs/methygpt`)
- **GPU**: NVIDIA H100 NVL (`CUDA_VISIBLE_DEVICES=1`)
- **Python**: 3.9.21
- **Reference repo**: `/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT` (read-only)
