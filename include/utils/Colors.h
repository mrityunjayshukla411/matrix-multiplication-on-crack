#pragma once

namespace Colors {
    // Reset
    constexpr const char* RESET = "\033[0m";

    // Regular colors
    constexpr const char* BLACK = "\033[0;30m";
    constexpr const char* RED = "\033[0;31m";
    constexpr const char* GREEN = "\033[0;32m";
    constexpr const char* YELLOW = "\033[0;33m";
    constexpr const char* BLUE = "\033[0;34m";
    constexpr const char* MAGENTA = "\033[0;35m";
    constexpr const char* CYAN = "\033[0;36m";
    constexpr const char* WHITE = "\033[0;37m";

    // Bold colors
    constexpr const char* BOLD_BLACK = "\033[1;30m";
    constexpr const char* BOLD_RED = "\033[1;31m";
    constexpr const char* BOLD_GREEN = "\033[1;32m";
    constexpr const char* BOLD_YELLOW = "\033[1;33m";
    constexpr const char* BOLD_BLUE = "\033[1;34m";
    constexpr const char* BOLD_MAGENTA = "\033[1;35m";
    constexpr const char* BOLD_CYAN = "\033[1;36m";
    constexpr const char* BOLD_WHITE = "\033[1;37m";

    // Background colors
    constexpr const char* BG_BLACK = "\033[40m";
    constexpr const char* BG_RED = "\033[41m";
    constexpr const char* BG_GREEN = "\033[42m";
    constexpr const char* BG_YELLOW = "\033[43m";
    constexpr const char* BG_BLUE = "\033[44m";
    constexpr const char* BG_MAGENTA = "\033[45m";
    constexpr const char* BG_CYAN = "\033[46m";
    constexpr const char* BG_WHITE = "\033[47m";

    // Text styles
    constexpr const char* BOLD = "\033[1m";
    constexpr const char* DIM = "\033[2m";
    constexpr const char* ITALIC = "\033[3m";
    constexpr const char* UNDERLINE = "\033[4m";
    constexpr const char* BLINK = "\033[5m";
    constexpr const char* REVERSE = "\033[7m";
}
