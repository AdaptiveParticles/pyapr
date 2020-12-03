import unittest
from numbers import Number
import pyapr


class ArithmeticTests(unittest.TestCase):

    def setUp(self):

        self.numel = 113

        self.float1 = pyapr.FloatParticles(self.numel)
        self.float2 = pyapr.FloatParticles(self.numel)
        self.short1 = pyapr.ShortParticles(self.numel)
        self.short2 = pyapr.ShortParticles(self.numel)

        self.float1.fill(1)
        self.float2.fill(1/3)
        self.short1.fill(2)
        self.short2.fill(3)

        self.eps = 1e-6

    def __test_outofplace_operation(self, in1, in2, op):
        tmp = op(in1, in2)

        if not any(isinstance(p, (pyapr.FloatParticles, pyapr.ShortParticles)) for p in [in1, in2]):
            return False

        # if all inputs are particles
        if all(isinstance(p, (pyapr.FloatParticles, pyapr.ShortParticles)) for p in [in1, in2]):
            # check output datatype
            if all(isinstance(p, pyapr.ShortParticles) for p in [in1, in2]):
                self.assertTrue(isinstance(tmp, pyapr.ShortParticles))
            else:
                self.assertTrue(isinstance(tmp, pyapr.FloatParticles))

            # check result
            for (x, y, z) in zip(tmp, in1, in2):
                self.assertAlmostEqual(x, op(y, z), delta=self.eps)
        else:
            p = in1 if isinstance(in1, (pyapr.FloatParticles, pyapr.ShortParticles)) else in2
            c = in2 if isinstance(in1, (pyapr.FloatParticles, pyapr.ShortParticles)) else in1
            self.assertTrue(isinstance(c, Number))
            self.assertTrue(isinstance(p, (pyapr.FloatParticles, pyapr.ShortParticles)))

            # check result
            for (x, y) in zip(tmp, p):
                self.assertAlmostEqual(x, op(y, c), delta=self.eps)


    @staticmethod
    def __pluseq(a, b):
        a += b

    @staticmethod
    def __minuseq(a, b):
        a -= b

    @staticmethod
    def __timeseq(a, b):
        a *= b

    def __test_inplace_operation(self,
                                 in1: (pyapr.FloatParticles, pyapr.ShortParticles),
                                 in2: Number,
                                 op,
                                 op_valcheck):

        tmp = in1.copy()
        for (x, y) in zip(tmp, in1):
            self.assertEqual(x, y)

        op(tmp, in2)
        if isinstance(in2, (pyapr.FloatParticles, pyapr.ShortParticles)):
            for (x, y, z) in zip(tmp, in1, in2):
                self.assertAlmostEqual(x, op_valcheck(y, z), delta=self.eps)
        elif isinstance(in2, Number):
            for (x, y) in zip(tmp, in1):
                self.assertAlmostEqual(x, op_valcheck(y, in2), delta=self.eps)

    def test_init(self):
        for x in [self.float1, self.float2, self.short1, self.short2]:
            self.assertEqual(len(x), self.numel)

        for p in self.float2:
            self.assertAlmostEqual(p, 1/3, delta=self.eps)

        for p in self.short2:
            self.assertEqual(p, 3)

    def test_addition(self):
        self.__test_outofplace_operation(self.float1, self.float2, lambda x, y: x+y)
        self.__test_outofplace_operation(self.short1, self.short2, lambda x, y: x+y)
        self.__test_outofplace_operation(self.short1, self.float1, lambda x, y: x + y)
        self.__test_outofplace_operation(self.float2, self.short2, lambda x, y: x + y)
        self.__test_outofplace_operation(self.float1, 0.37, lambda x, y: x + y)
        self.__test_outofplace_operation(self.short1, 21, lambda x, y: x + y)

        self.__test_inplace_operation(self.float1, 1/3, self.__pluseq, lambda x, y: x+y)
        self.__test_inplace_operation(self.short1, 30, self.__pluseq, lambda x, y: x + y)
        self.__test_inplace_operation(self.float1, self.float2, self.__pluseq, lambda x, y: x + y)
        self.__test_inplace_operation(self.short1, self.short2, self.__pluseq, lambda x, y: x + y)
        self.__test_inplace_operation(self.float1, self.short2, self.__pluseq, lambda x, y: x + y)

    def test_subtraction(self):
        self.__test_outofplace_operation(self.float1, self.float2, lambda x, y: x - y)
        self.__test_outofplace_operation(self.short2, self.short1, lambda x, y: x - y)
        self.__test_outofplace_operation(self.short1, self.float1, lambda x, y: x - y)
        self.__test_outofplace_operation(self.float2, self.short2, lambda x, y: x - y)
        self.__test_outofplace_operation(self.float2, 0.13, lambda x, y: x - y)
        self.__test_outofplace_operation(self.short2, -1, lambda x, y: x - y)

        self.__test_inplace_operation(self.float1, 29.78, self.__minuseq, lambda x, y: x - y)
        self.__test_inplace_operation(self.short1, 1, self.__minuseq, lambda x, y: x - y)
        self.__test_inplace_operation(self.float1, self.float2, self.__minuseq, lambda x, y: x - y)
        self.__test_inplace_operation(self.short2, self.short1, self.__minuseq, lambda x, y: x - y)
        self.__test_inplace_operation(self.float1, self.short2, self.__minuseq, lambda x, y: x - y)

    def test_multiplication(self):
        self.__test_outofplace_operation(self.float1, self.float2, lambda x, y: x * y)
        self.__test_outofplace_operation(self.short2, self.short1, lambda x, y: x * y)
        self.__test_outofplace_operation(self.short1, self.float1, lambda x, y: x * y)
        self.__test_outofplace_operation(self.float2, self.short2, lambda x, y: x * y)
        self.__test_outofplace_operation(self.float2, 0.19, lambda x, y: x * y)
        self.__test_outofplace_operation(self.short2, 13, lambda x, y: x * y)
        self.__test_outofplace_operation(0.73, self.float1, lambda x, y: x * y)

        self.__test_inplace_operation(self.float1, 0.97, self.__timeseq, lambda x, y: x * y)
        self.__test_inplace_operation(self.short1, 10, self.__timeseq, lambda x, y: x * y)
        self.__test_inplace_operation(self.float1, self.float2, self.__timeseq, lambda x, y: x * y)
        self.__test_inplace_operation(self.short2, self.short1, self.__timeseq, lambda x, y: x * y)
        self.__test_inplace_operation(self.float1, self.short2, self.__timeseq, lambda x, y: x * y)


if __name__ == '__main__':
    unittest.main()
