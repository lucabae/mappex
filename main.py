"""Entry point: map the corridor and write a top-down map.

    python main.py                    # opens the camera and map windows
    python main.py --headless -o map.png
    python main.py --weights weights/room_detection.pt
"""

from mappex.pipeline import main

if __name__ == "__main__":
    main()
