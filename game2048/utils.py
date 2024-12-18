

def get_arrow_key() -> str:
    """Get arrow key input from the user"""
    try:
        import msvcrt  # Windows
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'\xe0':  # Special key prefix
                    key = msvcrt.getch()
                    if key == b'H': return 'up'
                    if key == b'P': return 'down'
                    if key == b'K': return 'left'
                    if key == b'M': return 'right'
                return key.decode().lower()
    except ImportError:
        # Unix-like systems
        try:
            import termios
            import sys, tty
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # Escape sequence
                    ch = sys.stdin.read(2)
                    if ch == '[A': return 'up'
                    if ch == '[B': return 'down'
                    if ch == '[D': return 'left'
                    if ch == '[C': return 'right'
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except ImportError:
            # Fallback for systems without termios
            return input("Enter move (W/A/S/D or Q to quit): ").lower()