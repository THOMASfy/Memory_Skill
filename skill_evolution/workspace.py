"""Copy S0 SKILL.md into a session workspace; destroy on close."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path


class SkillWorkspace:
    def __init__(self, baseline_skill_md: Path, session_dir: Path):
        self.baseline_skill_md = baseline_skill_md
        self.session_dir = session_dir
        self.skill_md = session_dir / "SKILL.md"
        self.state_path = session_dir / "evolution_state.json"

    @classmethod
    def open(cls, baseline_skill_md: Path, session_dir: Path | None = None) -> SkillWorkspace:
        baseline_skill_md = baseline_skill_md.resolve()
        if not baseline_skill_md.is_file():
            raise FileNotFoundError(baseline_skill_md)
        if session_dir is None:
            session_dir = Path(tempfile.mkdtemp(prefix="skill_evo_"))
        else:
            session_dir.mkdir(parents=True, exist_ok=True)
        ws = cls(baseline_skill_md, session_dir)
        shutil.copy2(baseline_skill_md, ws.skill_md)
        return ws

    def destroy(self) -> None:
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)
