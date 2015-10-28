from unittest import TestCase

from task4 import get_unit_count


class Task4TestCase(TestCase):

    def test_get_unit_count_2(self):
        unit_count = get_unit_count(2)
        actual = (784 * unit_count) + ((2-1)*(unit_count**2)) + (unit_count * 10)
        print int(unit_count)
        self.assertAlmostEqual(actual, 635200)

    def test_get_unit_count_3(self):
        unit_count = get_unit_count(3)
        print int(unit_count)
        actual = (784 * unit_count) + ((3-1)*(unit_count**2)) + (unit_count * 10)
        self.assertAlmostEqual(actual, 635200)

    def test_get_unit_count_4(self):
        unit_count = get_unit_count(4)
        print int(unit_count)
        actual = (784 * unit_count) + ((4-1)*(unit_count**2)) + (unit_count * 10)
        self.assertEqual(actual, 635200)

    def test_get_unit_count_5(self):
        unit_count = get_unit_count(5)
        print int(unit_count)
        actual = (784 * unit_count) + ((5-1)*(unit_count**2)) + (unit_count * 10)
        self.assertEqual(actual, 635200)