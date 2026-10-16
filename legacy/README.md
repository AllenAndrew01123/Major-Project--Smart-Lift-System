# Legacy (original 2025 project)

* `papers/` – review decks (Phase I, Phase II, ESE) and the research paper.
* `code/` – the original single-lift simulators and `C.py`.
  Note: `C.py` generated the power-consumption comparison from straight lines with guessed slopes
  plus noise. It was not measured from a simulation. The revamp replaces it with measured energy.
* `data/lift_calls.xlsx` – original synthetic call log (100k ticks).

The original project also used [fourjr/elevator-simulator](https://github.com/fourjr/elevator-simulator) as a reference.
It is not included here (no license); the revamp's simulator is written from scratch.
