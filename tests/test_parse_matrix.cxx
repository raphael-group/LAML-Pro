#include <iostream>
#include <vector>
#include <string>
#include "io.h"  

int main() {
    const std::string filename = "/Users/gc3045/laml2_experiments/BaseMEM_Magic/gillian/baseMemoir_data_clean_reformatted_pos2.csv";

    auto [taxa_names, character_keys, matrix] = parse_observation_matrix(filename);

    std::cout << "Taxa names:\n";
    for (const auto& name : taxa_names) {
        std::cout << "  " << name << "\n";
    }

    std::cout << "\nCharacter keys:\n";
    for (const auto& [cassette, target] : character_keys) {
        std::cout << "  (" << cassette << ", " << target << ")\n";
    }

    std::cout << "\nMatrix:\n";
    for (size_t i = 0; i < matrix.size(); ++i) {
        std::cout << "Taxon " << taxa_names[i] << ":\n";
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << "  Char " << j << ": [";
            for (size_t k = 0; k < matrix[i][j].size(); ++k) {
                std::cout << matrix[i][j][k];
                if (k < matrix[i][j].size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }

    return 0;
}

