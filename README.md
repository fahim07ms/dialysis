# Dialysis — DTMF Signal Analysis Studio

## Run it

```bash
pip install -r requirements.txt
python main.py
```

(Linux users: `sounddevice` needs PortAudio installed at the OS level —
`sudo apt install libportaudio2` if `python main.py` complains about it.)

## Where things live

```
main.py                 <- entry point: loads fonts, applies theme, launches window
core/
  dsp_interface.py       <- YOUR FILE. All signal-processing math goes here.
                            Every function is a stub with a TODO comment
                            explaining exactly what to build and its exact
                            input/output shape. Run it directly to self-test
                            without opening the GUI: `python core/dsp_interface.py`
  audio_io.py            <- playback/WAV read-write plumbing (already done, no need to touch)
ui/
  theme.py                <- all colors + the QSS stylesheet (Pixel-style dark/teal theme)
  keypad.py                <- the 4x4 keypad widget
  left_panel.py            <- keypad + text field + play/export/import controls
  right_panel.py           <- tabbed analysis area (placeholders — filled in phase by phase)
  main_window.py           <- assembles left + right panel into the split window
assets/fonts/            <- bundled Nunito variable font files
```

## The contract

- **`core/`** never imports PyQt. It only takes/returns numpy arrays, numbers, and strings.
- **`ui/`** never does math. It calls functions in `core/dsp_interface.py` and just displays
  whatever comes back.

## Status: Phase 0 + Phase 1 scaffolding complete

- ✅ App skeleton, two-pane splitter layout, Pixel-style dark/teal theme, Nunito font
- ✅ 4x4 keypad wired to `dsp.generate_dtmf_tone()` and `dsp.sequence_to_wav()`
- ✅ WAV export/import UI wired to `core/audio_io.py`
- ⏳ Waiting on you: implement `generate_dtmf_tone()` in `core/dsp_interface.py` — that's the
  ONLY function you need for tone playback to start working. Until then, pressing keys shows a
  friendly status message instead of crashing.

Next up once Phase 1 works: **Phase 2 (WAV export)**, already wired to `sequence_to_wav()` — just needs your implementation.
