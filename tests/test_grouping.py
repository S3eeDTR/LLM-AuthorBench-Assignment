import unittest
from src.prepare_data import canonical
class GroupingTests(unittest.TestCase):
    def test_unquoted_random_input(self):
        self.assertEqual(canonical('Generate a C program that creates a histogram from data: randomABC123.'),canonical('Generate a C program that creates a histogram from data: otherXYZ999.'))
    def test_semantic_alias(self):
        self.assertEqual(canonical('Create a C program that solves Sudoku puzzles using backtracking.'),canonical('Generate a C program that solves a 85x85 Sudoku puzzle.'))
    def test_filename_and_number(self):
        self.assertEqual(canonical("Write a C program to analyze word frequency in 'notes.md'."),canonical("Write a C function to analyze word frequency in 'other.txt'."))
