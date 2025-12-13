# Cosmos Control System — Controller–Compute Architecture

**Version**: 1.0  
**Author**: Fabian (concept), ChatGPT (formal spec)  
**Audience**: Codex (implementation)

## 1. System Overview
- **Controller Service**: brain and single source of truth on the main machine; owns slicing, storage, job/slice state, assignment, aggregation, and reporting under `data/science_runs/<execution_id>/`.
- **Compute Worker Nodes**: barebones agents that report core count, receive slices, execute them, and return results; no local state except cached datasets.

## 2. Design Principles
1. **Workers are dumb:** no slicing/aggregation/reporting/directory creation/logic; compute → result → fetch next.
2. **Controller owns everything:** slicing, storage, aggregation, reporting, monitoring, slice assignment, dataset updates, cancellation.
3. **Slot allocation:** remote nodes get `floor(cores * 0.8)` slots, local nodes `floor(cores * 0.5)` slots; controller tracks occupancy.
4. **Filesystem boundary:** workers never read/write controller folders; only controller writes to `data/science_runs/<execution_id>/`.
5. **Hash-based datasets:** workers receive dataset updates only when hashes change.

## 3. Job Model
- **Job**: identified by `execution_id`, includes config, slice pool, aggregate status/progress/results/report.
- **Slice**: smallest CPU-core-sized unit; independent with parameters, dataset range, metrics, and status metadata.

## 4. Controller Responsibilities
1. **Job Intake:** accept submissions (HTTP/internal API), validate via `ScienceRunConfig`, generate `execution_id`, instantiate `data/science_runs/<execution_id>/` directory structure.
2. **Job Slicing:** expand job into slices according to run type (jackknife, partitions, parameter sweeps); every slice self-contained.
3. **Slice Queue Management:** track pending, assigned, completed slices.
4. **Worker Slot Management:** compute remote/local slots per worker, monitor free vs busy.
5. **Slice Assignment:** when worker requests work, assign up to available slot count with slice descriptor, config, dataset subset instructions.
6. **Aggregation:** on completion merge logs, metrics, partial results; compute chi²/likelihoods; update job progress/state.
7. **Reporting:** write final HTML/JSON report into `data/science_runs/<execution_id>/report/`.

## 5. Internal Data Structures
### 5.1 JobRecord
```
execution_id: str, run_id: str, config: dict,
slices: Dict[slice_id, SliceRecord],
created_at, started_at?, ended_at?,
status: queued|running|completed|failed|canceled,
aggregate_progress: float
```
### 5.2 SliceRecord
```
slice_id, kind, index, total_slices,
dataset_id?, range?, parameters: dict,
status: pending|assigned|running|completed|failed|canceled,
assigned_node?, progress: float,
started_at?, ended_at?
```
Slices instantiated by controller, remain independent with clear lifecycle metadata.

## 6. Slice Descriptor
Controller sends:
```
{
  "slice_id": "jk_003_of_020",
  "kind": "jackknife",
  "index": 3,
  "total_slices": 20,
  "dataset_id": "pantheon_plus",
  "range": { "start": 0.10, "end": 0.15 },
  "parameters": { ... }
}
```
Descriptor must accompany every slice assignment.

## 7. Worker Responsibilities
- Connect to the controller, identify `worker_id` + core count, report dataset hashes.
- Receive dataset updates (hash mismatch) and cache locally.
- Request slices, execute, and return results without writing to controller storage.
- No slicing, aggregation, reporting, or run orchestration logic.

## 8. Worker Workflow
1. **Startup (`worker_hello`)**
```
{ worker_id, cores: N, datasets: {dataset_id: hash} }
```
Controller replies with dataset updates (if needed) and slot confirmations.
2. **Work loop**
   - Send `request_work` with current load.
   - Receive `JobAssignment` array (each: `execution_id`, `run_id`, `config`, `slice`).
   - Run CPU-bound compute per slice via internal compute function; catch exceptions.
   - Return per-slice `SliceCompletion`:
```
{ execution_id, slice_id, success, logs, metrics, data }
```
   - Repeat until told `no_work` or cancellation.

## 9. Dataset Sync Protocol
- Workers report `datasets: {dataset_id: hash}`; controller compares to canonical hash.
- On mismatch, controller replies with `dataset_update`: includes `dataset_id`, `hash`, `payload` bytes/chunks.
- Workers cache payload locally; they never access controller folders.

## 10. Cancellation
- Controller marks job `cancel_requested`; workers observe cancel flag when requesting slices or reporting progress.
- Workers stop fetching new slices (optional mid-run abort).
- Controller cancels pending slices and updates job state to `canceled` once all slices resolved.

## 11. Progress Logic
- Controller-derived aggregate progress: mean of `slice.progress` (0 if never started).
- Workers report slice progress; controller recalculates job-level progress.

## 12. Completion Rules
- **Completed:** all slices `completed`.
- **Failed:** any slice `failed` with no retry.
- **Canceled:** cancellation requested and all slices resolved.

## 13. Filesystem Layout (Controller Only)
```
data/science_runs/<execution_id>/
  config.json
  state.json
  logs.txt
  slices/<slice_id>/
    logs.txt
    result.json
    metrics.json
  report/
    index.html
    assets/...
```
Workers never write here.

## 14. Controller APIs for Web UI
1. `POST /controller/jobs` – submit job.
2. `GET /controller/jobs` – list jobs.
3. `GET /controller/jobs/{execution_id}` – job details.
4. `GET /controller/jobs/{execution_id}/slices/{slice_id}` – slice detail.
5. `GET /controller/jobs/{execution_id}/logs` – job logs.
6. `GET /controller/jobs/{execution_id}/slices/{slice_id}/logs` – slice logs.
7. `POST /controller/jobs/{execution_id}/cancel` – cancel job.
UI only talks to controller.

## 15. Controller–Worker Protocol (`hpc_comms`)
### Incoming (worker → controller)
- `worker_hello`
- `request_work`
- `slice_progress`
- `slice_completion`
- `dataset_hash_summary`
- `worker_error`
### Outgoing (controller → worker)
- `dataset_update`
- `job_assignment`
- `cancel_slice`
- `cancel_all`
- `no_work`
Leverage existing serialization/transport primitives in `hpc_comms/`.

## 16. Required Invariants
1. Workers never access controller filesystem.
2. Controller alone writes run data.
3. Workers only compute and return structured results.
4. Slices are independent units.
5. Controller can reassign slices if worker dies.
6. Controller withstands partial failures.
7. UI interfaces solely with controller HTTP.

## 17. Implementation Order
1. **Worker hello & slot system:** handle `worker_id`, cores, dataset hashes, slot allocation (remote 80%, local 50%).
2. **Slice model:** descriptor, queue, assignment logic.
3. **Worker loop:** `hello`, `request_work`, `job_assignment`, `slice_completion`, repeat.
4. **Controller aggregation:** write slice folders, aggregate results/state.
5. **Job lifecycle:** queued → running → completed (including cancellation/failure).
6. **Dataset sync:** compare hashes, send updates.
7. **Report generation:** HTML in `report/`.
8. **UI endpoints:** job listing, detail, logs, slices, cancel.

## 18. Prohibitions
- No worker-side slicing or multi-slice logic.
- Workers may not write to controller directories.
- Controller/worker messaging must stay within `hpc_comms`.
- No direct worker ↔ UI communications.
- Keep controller and local compute logic separate; workers remain pure compute units.
