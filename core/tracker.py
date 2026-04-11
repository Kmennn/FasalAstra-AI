# PURPOSE: Anti-Double-Trigger System
# WHY THIS EXISTS:
#   Camera runs at 30 FPS
#   Same weed appears in 5-10 consecutive frames
#   Without tracking → fires solenoid 5-10 times on same weed
#   = massive chemical waste = defeats entire purpose
#
# HOW IT WORKS:
#   Each detected weed gets a unique ID based on its position
#   Once a timer is set for that weed → it's "locked"
#   Locked weeds are ignored in all future frames
#   Lock expires after 2 seconds (weed is now behind farmer)

import time
import math

LOCK_DURATION_S  = 2.0    # how long to ignore same weed
POSITION_TOLERANCE = 30   # pixels — how close = same weed

class WeedTracker:
    def __init__(self):
        self.active_weeds = {}
        # Format: { weed_id: { cx, cy, locked_at } }
        self.next_id = 0

    def _find_existing(self, cx, cy):
        """Check if this position matches a tracked weed"""
        for wid, data in self.active_weeds.items():
            dx = abs(data['cx'] - cx)
            dy = abs(data['cy'] - cy)
            dist = math.sqrt(dx**2 + dy**2)
            if dist < POSITION_TOLERANCE:
                return wid
        return None

    def _cleanup_expired(self):
        """Remove weeds that have passed behind the farmer"""
        now = time.time()
        expired = [
            wid for wid, data in self.active_weeds.items()
            if now - data['locked_at'] > LOCK_DURATION_S
        ]
        for wid in expired:
            del self.active_weeds[wid]

    def should_fire(self, cx, cy):
        """
        Check if we should fire for this weed position
        Returns: (should_fire, weed_id)
        """
        self._cleanup_expired()

        existing_id = self._find_existing(cx, cy)

        if existing_id is not None:
            # Weed already tracked — SKIP (anti double trigger)
            return False, existing_id

        # New weed — assign ID and lock it
        weed_id = self.next_id
        self.next_id += 1

        self.active_weeds[weed_id] = {
            'cx': cx,
            'cy': cy,
            'locked_at': time.time()
        }

        return True, weed_id

    def get_active_count(self):
        self._cleanup_expired()
        return len(self.active_weeds)


# Quick test
if __name__ == '__main__':
    tracker = WeedTracker()

    print("Anti-Double-Trigger Test:")
    print("-" * 40)

    fire1, id1 = tracker.should_fire(160, 200)
    print(f"  Frame 1 - Weed at (160,200): {'FIRE' if fire1 else 'SKIP'} ID={id1}")

    fire2, id2 = tracker.should_fire(162, 198)
    print(f"  Frame 2 - Same weed (162,198): {'FIRE' if fire2 else 'SKIP'} ID={id2}")

    fire3, id3 = tracker.should_fire(50, 100)
    print(f"  Frame 3 - New weed (50,100): {'FIRE' if fire3 else 'SKIP'} ID={id3}")

    print(f"\n  Active tracked weeds: {tracker.get_active_count()}")
