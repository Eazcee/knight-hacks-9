-- Street view: which floor each place is on (barometer / "floor 2" by voice).
ALTER TABLE nodes ADD COLUMN IF NOT EXISTS floor INTEGER DEFAULT 1;
