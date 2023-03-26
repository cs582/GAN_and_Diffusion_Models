import unittest
from src.tests.unit_tests.model_testing import TestMicroCNN, TestBuildingBlocks, TestMiniCNN, TestMNISTGen, TestDifussionModels

TestMicroCNN()
TestBuildingBlocks()
TestMiniCNN()
TestMNISTGen()
TestDifussionModels()

unittest.main()