# HNSWonGPUs design docs

One document per phase. The [roadmap](../roadmap.md) is the plan written before building; these are the designs as built, kept current while the phase is in progress and frozen when it is done — except for **post-close amendments**: an owner decision landed after a phase closes (so far: the worker spread rule and the draining scheduler, decisions 39 and 40) is added to the closed doc as an amendment section with its own cross-check, and the decisions table grows a row, so the table stays the one place every decision lives. Significant bugs are written up in the [bug log](../bugs.md); every test and its scenario is in the [test catalog](../tests.md).

| Doc | Status |
| --- | --- |
| [Phase 1: scheduler with fake jobs](phase1-scheduler.md) | Done: decisions 1–68, bugs 1–8 |
| [Phase 2: Kubernetes on kind](phase2-kubernetes.md) | Done: decisions 1–40, bugs 9–10, plus two post-close amendments |
| [Phase 3: placement policies and experiments](phase3-policies.md) | Next; not started |
| [Phase 4: real CPU builds with hnswlib](phase4-hnswlib.md) | Planned |
| [Phase 5: GPUs and cuVS](phase5-gpu.md) | Planned |
| [Phase 6: Kubernetes-native dispatch](phase6-kubernetes-native.md) | Optional, planned |

Decision numbers restart per phase: "decision 19" means different things in the Phase 1 and Phase 2 docs, so cite decisions together with their phase outside these files.

## Template

Every doc has the same sections, so a reader knows where to look.

1. **Status and scope.** What this phase builds and the done-when condition.
2. **System diagram.** Components, where they run, what talks to what.
3. **Flows.** Sequence or state diagrams for the paths that matter, including failure paths.
4. **Design decisions.** A table: decision, alternatives considered, why. Append-only; a reversed decision gets a new row that points at the old one.
5. **Implementation notes.** Filled in as components land: file layout, commands, anything a reader needs to run or modify it. Each step's notes are followed by the notes of its independent cross-check round, findings triaged as bug, spec gap or test error.
6. **Open questions.** Things decided later or deferred to a later phase; a question the owner answers is marked decided with its decision number, never silently deleted.

Diagrams are Mermaid in fenced code blocks, which GitHub and Obsidian render inline.
