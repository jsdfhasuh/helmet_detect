from helmet_detect.config import TemporalConfig
from helmet_detect.temporal import TemporalAlarm


def test_temporal_alarm_requires_multiple_hits_and_triggers_once() -> None:
    alarm = TemporalAlarm(TemporalConfig(window=5, min_hits=2, cooldown_seconds=3))
    assert not alarm.update(0.0, True).event_triggered
    second = alarm.update(0.2, True)
    assert second.event_triggered
    assert second.active
    assert not alarm.update(0.4, True).event_triggered


def test_temporal_alarm_rearms_after_clear_window_and_cooldown() -> None:
    alarm = TemporalAlarm(TemporalConfig(window=3, min_hits=2, cooldown_seconds=1))
    alarm.update(0.0, True)
    assert alarm.update(0.1, True).event_triggered
    alarm.update(0.2, False)
    alarm.update(0.3, False)
    alarm.update(0.4, False)
    alarm.update(1.2, True)
    assert alarm.update(1.3, True).event_triggered
