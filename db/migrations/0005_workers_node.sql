-- Phase 3 step 2 (decision 11): which machine a worker runs on. Workers on
-- one node share its network link, so the queue-replay estimator needs the
-- grouping to model concurrent transfers. Empty means unknown; the estimator
-- then treats the worker as alone on its own link.
ALTER TABLE workers ADD COLUMN node text NOT NULL DEFAULT '';
