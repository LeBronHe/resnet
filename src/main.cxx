#include <iostream>
#include <string_view>
#include <vector>

#include "mnist.hxx"

auto main(int argc, char** argv) -> int {
    std::vector<std::string_view> args(argv + 1, argv + argc);

    for (const auto& arg : args) {
        std::cout << arg << std::endl;
    }

    return 0;
}
