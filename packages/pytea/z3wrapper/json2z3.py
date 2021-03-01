#!/usr/bin/env python

"""
json2z3.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Se Hoon Kim

Encode json-formatted constraints to z3 formula, solve it,
and return constraint conditions.
"""

from z3 import *
from enum import Enum
from functools import reduce
from pathlib import Path
import json
import sys
import time


# z3 doesn't care of division by zero.
# TODO: (on constraintGenerator)Add constraint(divisor != 0) for every div/mod op.
def z3_div(a, b):
    return If(b != 0, a / b, -1)


def z3_mod(a, b):
    return If(b != 0, a % b, -1)


def z3_min(a, b):
    return If(b < a, b, a)


def z3_max(a, b):
    return If(a < b, b, a)


class PathResult(Enum):
    Unavailable = 0
    Valid = 1
    Sat = 2
    Unsat = 3
    DontKnow = 4


# Type enums: Must be consistent with enums in "symExpressions.ts", "constraintType.ts".
class SEType(Enum):
    Shape = 0
    Num = 1
    Bool = 2
    String = 3


class BoolOpType(Enum):
    Const = 0
    Symbol = 1
    Equal = 2
    NotEqual = 3
    LessThan = 4
    LessThanOrEqual = 5
    Not = 6
    And = 7
    Or = 8


class NumOpType(Enum):
    Const = 0
    Symbol = 1
    Bop = 2
    Index = 3
    Max = 4
    Numel = 5
    Uop = 6
    Min = 7


class NumBopType(Enum):
    Add = 0
    Sub = 1
    Mul = 2
    TrueDiv = 3
    FloorDiv = 4
    Mod = 5


class NumUopType(Enum):
    Neg = 0
    Floor = 1
    Ceil = 2
    Abs = 3


class ShapeOpType(Enum):
    Const = 0
    Symbol = 1
    Set = 2
    Slice = 3
    Concat = 4
    Broadcast = 5


class ConstraintType(Enum):
    ExpBool = 0
    Equal = 1
    NotEqual = 2
    And = 3
    Or = 4
    Not = 5
    LessThan = 6
    LessThanOrEqual = 7
    Forall = 8
    Broadcastable = 9
    Fail = 10


class Z3Encoder:
    def __init__(self, console):
        """
        the structure of json file should be like below.
        ```json
        [
            { ... }, # ctrSet of path_0,
            { ... }, # ctrSet of path_1,
              ...,
            { ... }  # ctrSet of path_n-1,
        ]
        ```

        Parameters:
            console: pipe to print logs. it should have `.log` method.
        """
        self.console = console

    def analyze(self, jsonObj):
        ctrSetList = map(CtrSet, jsonObj)

        # lists of path indices
        UnavailablePaths = []
        ValidPaths = []
        SatPaths = []
        UnsatPaths = []
        DontknowPaths = []

        for pathIdx, ctrSet in enumerate(ctrSetList):
            # comment out printing all constraints

            # log += ctrSet.toString()
            pathResult, pathLog, _ = ctrSet.analysis()  # side effect: print result
            # log += pathLog
            log = f"--- PATH {pathIdx} ---\n{pathLog}"

            if pathResult == PathResult.Unavailable.value:
                UnavailablePaths.append(pathIdx)
            elif pathResult == PathResult.Valid.value:
                ValidPaths.append(pathIdx)
            elif pathResult == PathResult.Sat.value:
                SatPaths.append(pathIdx)
                self.console.log(log)
            elif pathResult == PathResult.Unsat.value:
                UnsatPaths.append(pathIdx)
                self.console.log(log)
            else:
                DontknowPaths.append(pathIdx)
                self.console.log(log)

        self.console.log(
            "-----------------------------------\n"
            f"OVERALL (total {len(jsonObj)} paths)\n"
            "-----------------------------------"
        )
        valid_paths_len = len(ValidPaths) + len(SatPaths)
        if valid_paths_len != 0:
            self.console.log(f"Valid paths (no constraint error): {valid_paths_len}")
        if len(UnsatPaths) != 0:
            self.console.log(
                f"Invalid paths (found conflicted constraints): {len(UnsatPaths)}"
            )
        if len(DontknowPaths) != 0:
            self.console.log(f"Undecidable paths (z3 timeout): {len(DontknowPaths)}")
        if len(UnavailablePaths) != 0:
            self.console.log(
                f"Unreachable paths (conflicted branch conditions): {len(UnavailablePaths)}"
            )


# constraint set of a path.
class CtrSet:
    def __init__(self, jsonCtrSet):
        """
        ctrPool: list of constraints
        hardCtr: indices of ctrPool, whose ctr cannot be violated.
        softCtr: indices of ctrPool, whose ctr can be violated.
        pathCtr: indices of ctrPool, whose ctr indicates path conditions.
        """
        self.ctrPool = [Ctr(c) for c in jsonCtrSet["ctrPool"]]
        self.hardIdx = jsonCtrSet["hardCtr"]
        self.softIdx = jsonCtrSet["softCtr"]
        self.pathIdx = jsonCtrSet["pathCtr"]

        self.assumptions = [self.ctrPool[i].formula for i in self.hardIdx]
        self.pathCtrs = [self.ctrPool[i].formula for i in self.pathIdx]
        self.softCtrs = [self.ctrPool[i].formula for i in self.softIdx]

    def toString(self):
        assumptions = [self.ctrPool[i].formula for i in self.hardIdx]
        pathCtrs = [self.ctrPool[i].formula for i in self.pathIdx]
        softCtrs = [self.ctrPool[i].formula for i in self.softIdx]

        log = ""

        if len(assumptions) > 0:
            log += "Assumptions:\n"
            for assump in assumptions:
                log += assump.toString() + "\n"
            log += "\n"
        if len(pathCtrs) > 0:
            log += "Path Constraints:\n"
            for ctr in pathCtrs:
                log += ctr.toString() + "\n"
            log += "\n"
        if len(softCtrs) > 0:
            log += "Soft Constraints:\n"
            for ctr in softCtrs:
                log += ctr.toString() + "\n"
            log += "\n"

        return log

    # analyze ctrSet(of a path).
    def analysis(self):
        extras = dict()

        if self.pathCondCheck() == "unsat":
            log = "Unreachable path: Conflicted branch conditions."
            return PathResult.Unavailable.value, log, extras

        validity = self.checkValidity()
        if validity == "valid":
            log = "Valid path: Constraints are satisfiable."
            return PathResult.Valid.value, log, extras

        sat, unsatIndice = self.checkSat()
        if sat == PathResult.Sat.value:
            log = "Valid path: Constraints are satisfiable."
            return PathResult.Valid.value, log, extras
        elif sat == PathResult.Unsat.value:
            log = "Invalid path: Found conflicted constraints.\n\n"
            log += "first conflicted hard constraint:\n"
            log += self.ctrPool[unsatIndice].toString() + "\n"
            extras["conflict"] = unsatIndice
        else:
            sat = PathResult.DontKnow.value
            log = "Undecidable path: Z3 failed to solve constraints.\n\n"
            log += "first undecidable hard constraint:\n"
            log += self.ctrPool[unsatIndice].toString() + "\n"
            extras["undecide"] = unsatIndice

        return sat, log, extras

    # check sat with only hardCtr and pathCtr.
    def pathCondCheck(self):
        formula = And(self.pathCtrs)

        s = Solver()
        s.add(formula)

        return str(s.check(self.assumptions))

    # check validity and find counter-example if invalid.
    # return (validity, counter-example).
    def checkValidity(self):
        assumptions = self.assumptions + self.pathCtrs
        constraints = self.softCtrs
        formula = Implies(And(assumptions), And(constraints))
        neg = Not(formula)

        s = Solver()
        s.add(neg)
        if str(s.check()) == "unsat":
            return "valid"
        else:
            return "invalid"

    def checkSat(self, minimize=False):
        s = Solver()
        last_hard_idx = 0

        for curr_hard_idx in self.hardIdx:
            curr_list = self.ctrPool[last_hard_idx:curr_hard_idx]
            curr_hard = self.ctrPool[curr_hard_idx]
            s.add(And(And(curr_list), Not(curr_hard)))

            result = s.check()
            if result == "sat":
                return PathResult.Unsat.value, curr_hard_idx
            elif result == "unsat":
                pass
            else:
                return PathResult.DontKnow.value, curr_hard_idx

            last_hard_idx = curr_hard_idx

        return PathResult.Sat.value, None


class Ctr:
    def __init__(self, jsonCtr):
        self.json = jsonCtr
        self.formula = self.encode(jsonCtr)
        if "source" in jsonCtr:
            self.source = jsonCtr["source"]
        else:
            self.source = None

    def toString(self):
        ctrLog = str(self.formula)

        if self.source != None:
            start = self.source["start"]
            length = self.source["length"]
            sourceLog = " - [{}:{}]".format(str(start), str(start + length - 1))
            ctrLog += sourceLog

        return ctrLog

    def encode(self, ctr):
        if ctr["type"] == ConstraintType.ExpBool.value:
            return self.encodeExpBool(ctr["exp"])
        if ctr["type"] == ConstraintType.Equal.value:
            return self._encodeEq(ctr)
        elif ctr["type"] == ConstraintType.NotEqual.value:
            return self._encodeNe(ctr)
        elif ctr["type"] == ConstraintType.And.value:
            return self._encodeAnd(ctr)
        elif ctr["type"] == ConstraintType.Or.value:
            return self._encodeOr(ctr)
        elif ctr["type"] == ConstraintType.Not.value:
            return self._encodeNot(ctr)
        elif ctr["type"] == ConstraintType.LessThan.value:
            return self._encodeLt(ctr)
        elif ctr["type"] == ConstraintType.LessThanOrEqual.value:
            return self._encodeLe(ctr)
        elif ctr["type"] == ConstraintType.Forall.value:
            return self._encodeFa(ctr)
        elif ctr["type"] == ConstraintType.Broadcastable.value:
            return self._encodeBc(ctr)
        elif ctr["type"] == ConstraintType.Fail.value:
            return False

    def _encodeEq(self, ctr):
        left = ctr["left"]
        right = ctr["right"]
        if left["expType"] == SEType.Num.value and right["expType"] == SEType.Num.value:
            return self.encodeExpNum(left) == self.encodeExpNum(right)
        elif (
            left["expType"] == SEType.Shape.value
            and right["expType"] == SEType.Shape.value
        ):
            return self.encodeExpShape(left) == self.encodeExpShape(right)
        else:
            raise Exception("encodeConstraint Error: Comparison mismatch")

    def _encodeNe(self, ctr):
        left = ctr["left"]
        right = ctr["right"]
        if left["expType"] == SEType.Num.value and right["expType"] == SEType.Num.value:
            return self.encodeExpNum(left) != self.encodeExpNum(right)
        elif (
            left["expType"] == SEType.Shape.value
            and right["expType"] == SEType.Shape.value
        ):
            return self.encodeExpShape(left) != self.encodeExpShape(right)

    def _encodeAnd(self, ctr):
        left = self.encode(ctr["left"])
        right = self.encode(ctr["right"])
        return And(left, right)

    def _encodeOr(self, ctr):
        left = self.encode(ctr["left"])
        right = self.encode(ctr["right"])
        return Or(left, right)

    def _encodeNot(self, ctr):
        _ctr = self.encode(ctr["constraint"])
        return Not(_ctr)

    def _encodeLt(self, ctr):
        left = self.encodeExpNum(ctr["left"])
        right = self.encodeExpNum(ctr["right"])
        return left < right

    def _encodeLe(self, ctr):
        left = self.encodeExpNum(ctr["left"])
        right = self.encodeExpNum(ctr["right"])
        return left <= right

    def _encodeFa(self, ctr):
        """
        forall x in [lb, ub].<_ctr>
        """
        x = Int(ctr["symbol"]["name"])
        lb, ub = ctr["range"]
        lb, ub = self.encodeExpNum(lb), self.encodeExpNum(ub)
        if not (is_int(lb) and is_int(ub)):
            raise Exception("_encodeFa: both bounds must be ints")
        _ctr = self.encode(ctr["constraint"])
        return ForAll([x], Implies(And(lb <= x, x <= ub), _ctr))

    def _encodeBc(self, ctr):
        """
        BroadCastable [d0, d1,..,dn] [d0', d1',...,dn']:
            forall i in [0, n].(di == di' || di == 1 || di' == 1)
        """
        left = self.encodeExpShape(ctr["left"])
        right = self.encodeExpShape(ctr["right"])
        rankLeft = self.getRank(ctr["left"])
        rankRight = self.getRank(ctr["right"])
        i = Int("i")
        return Or(
            And(
                rankLeft >= rankRight,
                ForAll(
                    [i],
                    Implies(
                        And(rankLeft - rankRight <= i, i < rankLeft),
                        Or(
                            Select(left, i)
                            == Select(right, i - (rankLeft - rankRight)),
                            Select(left, i) == 1,
                            Select(right, i - (rankLeft - rankRight)) == 1,
                        ),
                    ),
                ),
            ),
            And(
                rankLeft < rankRight,
                ForAll(
                    [i],
                    Implies(
                        And(rankRight - rankLeft <= i, i < rankRight),
                        Or(
                            Select(right, i)
                            == Select(left, i - (rankRight - rankLeft)),
                            Select(right, i) == 1,
                            Select(left, i - (rankRight - rankLeft)) == 1,
                        ),
                    ),
                ),
            ),
        )

    def encodeExp(self, exp):
        if exp["expType"] == SEType.Num.value:
            return self.encodeExpNum(exp)
        elif exp["expType"] == SEType.Shape.value:
            return self.encodeExpShape(exp)
        elif exp["expType"] == SEType.String.value:
            raise Exception("encodeExp Error: not supported")
        elif exp["expBool"] == SEType.Bool.value:
            return self.encodeExpBool(exp)

    def encodeExpBool(self, expBool):
        if expBool["expType"] != SEType.Bool.value:
            raise Exception("encodeExpBool Error: not a ExpBool")

        if expBool["opType"] == BoolOpType.Const.value:
            return expBool["value"]
        if expBool["opType"] == BoolOpType.Symbol.value:
            return Bool(expBool["symbol"]["name"])
        if expBool["opType"] == BoolOpType.Equal.value:
            if expBool["left"]["expType"] != expBool["right"]["expType"]:
                raise Exception("encodeExpBool Error: comparison type mismatch")
            left = self.encodeExp(expBool["left"])
            right = self.encodeExp(expBool["right"])
            return left == right
        if expBool["opType"] == BoolOpType.NotEqual.value:
            if expBool["left"]["expType"] != expBool["right"]["expType"]:
                raise Exception("encodeExpBool Error: comparison type mismatch")
            left = self.encodeExp(expBool["left"])
            right = self.encodeExp(expBool["right"])
            return left != right
        if expBool["opType"] == BoolOpType.LessThan.value:
            if (
                expBool["left"]["expType"] != SEType.Num.value
                or expBool["right"]["expType"] != SEType.Num.value
            ):
                raise Exception("encodeExpBool Error: comparison type mismatch")
            left = self.encodeExpNum(expBool["left"])
            right = self.encodeExpNum(expBool["right"])
            return left < right
        if expBool["opType"] == BoolOpType.LessThanOrEqual.value:
            if (
                expBool["left"]["expType"] != SEType.Num.value
                or expBool["right"]["expType"] != SEType.Num.value
            ):
                raise Exception("encodeExpBool Error: comparison type mismatch")
            left = self.encodeExpNum(expBool["left"])
            right = self.encodeExpNum(expBool["right"])
            return left <= right
        if expBool["opType"] == BoolOpType.Not.value:
            baseBool = self.encodeExpBool(expBool["baseBool"])
            return Not(baseBool)
        if expBool["opType"] == BoolOpType.And.value:
            left = self.encodeExpBool(expBool["left"])
            right = self.encodeExpBool(expBool["right"])
            return And(left, right)
        if expBool["opType"] == BoolOpType.Or.value:
            left = self.encodeExpBool(expBool["left"])
            right = self.encodeExpBool(expBool["right"])
            return Or(left, right)

    def getRank(self, expShape):
        if expShape["expType"] != SEType.Shape.value:
            raise Exception("getRank Error: not a ExpShape")

        if expShape["opType"] == ShapeOpType.Const.value:
            return self.encodeExpNum(expShape["rank"])
        elif expShape["opType"] == ShapeOpType.Symbol.value:
            rank = self.encodeExpNum(expShape["symbol"]["rank"])
            if not is_int(rank):
                raise Exception("getRank(Symbol): a rank must be an int")
            return rank
        elif expShape["opType"] == ShapeOpType.Set.value:
            return self.getRank(expShape["baseShape"])
        elif expShape["opType"] == ShapeOpType.Slice.value:
            return self.encodeExpNum(expShape["end"]) - self.encodeExpNum(
                expShape["start"]
            )
        elif expShape["opType"] == ShapeOpType.Concat.value:
            return self.getRank(expShape["left"]) + self.getRank(expShape["right"])
        elif expShape["opType"] == ShapeOpType.Broadcast.value:
            rankLeft = self.getRank(expShape["left"])
            rankRight = self.getRank(expShape["right"])
            return z3_max(rankLeft, rankRight)

    def encodeExpNum(self, expNum):
        if expNum["expType"] != SEType.Num.value:
            raise Exception("encodeExpNum Error: not a ExpNum")

        if expNum["opType"] == NumOpType.Const.value:
            value = expNum["value"]
            if isinstance(value, int):
                return IntVal(value)
            elif isinstance(value, float):
                return RealVal(value)
            else:
                raise Exception(
                    "encodeExpNum(Const) Error: type of value must be int or float"
                )
        elif expNum["opType"] == NumOpType.Symbol.value:
            return Int(expNum["symbol"]["name"])
        elif expNum["opType"] == NumOpType.Bop.value:
            return self._encodeExpNumBop(expNum)
        elif expNum["opType"] == NumOpType.Index.value:
            baseShape = self.encodeExpShape(expNum["baseShape"])
            index = self.encodeExpNum(expNum["index"])
            if not is_int(index):
                raise Exception("encodeExpNum(Index) Error: index must be an int")
            return Select(baseShape, index)
        elif expNum["opType"] == NumOpType.Max.value:
            values = expNum["values"]
            encodedValues = map(lambda v: self.encodeExpNum(v), values)
            return reduce(lambda a, b: z3_max(a, b), encodedValues)
        elif expNum["opType"] == NumOpType.Numel.value:
            return self._encodeExpNumNumel(expNum)
        elif expNum["opType"] == NumOpType.Uop.value:
            return self._encodeExpNumUop(expNum)
        elif expNum["opType"] == NumOpType.Min.value:
            values = expNum["values"]
            encodedValues = map(lambda v: self.encodeExpNum(v), values)
            return reduce(lambda a, b: z3_min(a, b), encodedValues)

    def _encodeExpNumBop(self, expNum):
        if expNum["bopType"] == NumBopType.Add.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            return left + right
        elif expNum["bopType"] == NumBopType.Sub.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            return left - right
        elif expNum["bopType"] == NumBopType.Mul.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            return left * right
        elif expNum["bopType"] == NumBopType.TrueDiv.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            if is_int(left):
                left = ToReal(left)
            elif is_int(right):
                right = ToReal(right)
            return z3_div(left, right)
        elif expNum["bopType"] == NumBopType.FloorDiv.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            if is_real(left) or is_real(right):
                raise Exception(
                    "_encodeExpBop(FloorDiv) Error: both numbers must be Ints"
                )
            return z3_div(left, right)
        elif expNum["bopType"] == NumBopType.Mod.value:
            left = self.encodeExpNum(expNum["left"])
            right = self.encodeExpNum(expNum["right"])
            if is_real(left) or is_real(right):
                raise Exception("_encodeExpBop(Mod) Error: both numbers must be Ints")
            return z3_mod(left, right)

    def _encodeExpNumNumel(self, expNum):
        baseShape = expNum["shape"]
        baseShapeEncoded = self.encodeExpShape(baseShape)

        prod = RecFunction(
            "prod", ArraySort(IntSort(), IntSort()), IntSort(), IntSort(), IntSort()
        )
        shape = Array("shape", IntSort(), IntSort())
        lb = Int("lb")
        ub = Int("ub")
        RecAddDefinition(
            prod,
            (shape, lb, ub),
            If(lb > ub, 1, Select(shape, lb) * (prod(shape, (lb + 1), ub))),
        )

        return prod(baseShapeEncoded, 0, self.getRank(baseShape) - 1)

    def _encodeExpNumUop(self, expNum):
        if expNum["uopType"] == NumUopType.Neg.value:
            baseValue = self.encodeExpNum(expNum["baseValue"])
            return -baseValue
        elif expNum["uopType"] == NumUopType.Floor.value:
            baseValue = self.encodeExpNum(expNum["baseValue"])
            return baseValue if not is_real(baseValue) else ToInt(baseValue)
        elif expNum["uopType"] == NumUopType.Ceil.value:
            baseValue = self.encodeExpNum(expNum["baseValue"])
            if is_real(baseValue):
                floor = ToInt(baseValue)
                return If(floor == baseValue, floor, floor + 1)
            return baseValue
        elif expNum["uopType"] == NumUopType.Abs.value:
            baseValue = self.encodeExpNum(expNum["baseValue"])
            return If(baseValue < 0, -baseValue, baseValue)

    def encodeExpShape(self, expShape):
        if expShape["expType"] != SEType.Shape.value:
            raise Exception("encodeExpShape Error: not a ExpShape")

        # returns a z3 array
        if expShape["opType"] == ShapeOpType.Const.value:
            dims = expShape["dims"]
            shape = K(IntSort(), -1)
            for i in range(len(dims)):
                dim = self.encodeExpNum(dims[i])
                if not is_int(dim):
                    raise Exception("encodeExpShapa(Const): a dimension must be an int")
                shape = Store(shape, i, dim)
            return shape
        elif expShape["opType"] == ShapeOpType.Symbol.value:
            name = expShape["symbol"]["name"]
            rank = self.encodeExpNum(expShape["symbol"]["rank"])
            if not is_int(rank):
                raise Exception("encdoeExpShape(Symbol): a rank must be an int")
            shape = Array(name, IntSort(), IntSort())
            i = Int("i")
            return Lambda([i], If(And(0 <= i, i < rank), Select(shape, i), -1))
        elif expShape["opType"] == ShapeOpType.Set.value:
            return self._encodeExpShapeSet(expShape)
        elif expShape["opType"] == ShapeOpType.Slice.value:
            return self._encodeExpShapeSlice(expShape)
        elif expShape["opType"] == ShapeOpType.Concat.value:
            return self._encodeExpShapeConcat(expShape)
        elif expShape["opType"] == ShapeOpType.Broadcast.value:
            return self._encodeExpShapeBc(expShape)

    def _encodeExpShapeSet(self, expShape):
        baseShape = self.encodeExpShape(expShape["baseShape"])
        axis = self.encodeExpNum(expShape["axis"])
        if not is_int(axis):
            raise Exception("_encodeExpShapeSet: an axis must be an int")
        dim = self.encodeExpNum(expShape["dim"])
        if not is_int(dim):
            raise Exception("_encodeExpShapeSet: a dim must be an int")
        return Store(baseShape, axis, dim)

    def _encodeExpShapeSlice(self, expShape):
        dims = self.encodeExpShape(expShape["baseShape"])

        # TODO: How to handle cases where "start" and "end" are not given?
        start = self.encodeExpNum(expShape["start"])
        if start is None:
            start = 0
        if not is_int(start):
            raise Exception("_encodeExpShapeSlice: a start index must be an int")
        end = self.encodeExpNum(expShape["end"])
        if end is None:
            end = self.getRank(dims)
        if not is_int(end):
            raise Exception("_encodeExpShapeSlice: a end index must be an int")
        i = Int("i")
        return Lambda(
            [i], If(And(0 <= i, i < (end - start)), Select(dims, start + i), -1)
        )

    def _encodeExpShapeConcat(self, expShape):
        left = self.encodeExpShape(expShape["left"])
        rankLeft = self.getRank(expShape["left"])
        right = self.encodeExpShape(expShape["right"])
        rankRight = self.getRank(expShape["right"])
        i = Int("i")
        return Lambda(
            [i],
            If(
                And(0 <= i, i < rankLeft),
                Select(left, i),
                If(
                    And(rankLeft <= i, i < rankLeft + rankRight),
                    Select(right, i - rankLeft),
                    -1,
                ),
            ),
        )

    def _encodeExpShapeBc(self, expShape):
        left = self.encodeExpShape(expShape["left"])
        rankLeft = self.getRank(expShape["left"])
        right = self.encodeExpShape(expShape["right"])
        rankRight = self.getRank(expShape["right"])
        i = Int("i")
        return Lambda(
            [i],
            If(
                rankLeft >= rankRight,
                If(
                    And(0 <= i, i < rankLeft - rankRight),
                    Select(left, i),
                    If(
                        And(rankLeft - rankRight <= i, i < rankLeft),
                        z3_max(
                            Select(left, i), Select(right, i - (rankLeft - rankRight))
                        ),
                        -1,
                    ),
                ),
                If(
                    And(0 <= i, i < rankRight - rankLeft),
                    Select(right, i),
                    If(
                        And(rankRight - rankLeft <= i, i < rankRight),
                        z3_max(
                            Select(right, i), Select(left, i - (rankRight - rankLeft))
                        ),
                        -1,
                    ),
                ),
            ),
        )


class DefaultConsole:
    def log(self, message):
        print(message)


class NullConsole:
    def log(self, message):
        pass


def run_default(json_path, args):
    start_time = time.time()
    json_path = Path(json_path)

    console = DefaultConsole() if not args.silent else NullConsole()

    console.log("\n------------- z3 result -------------")

    if not json_path.exists():
        console.log(f"result json '{json_path}' does not exist")
        return

    with json_path.open("r") as f:
        ctr_set = json.load(f)
        encoder = Z3Encoder(console)
        encoder.analyze(ctr_set)

        console.log(f"z3 runtime: {time.time() - start_time:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./json2z3.py <json_file>")
        sys.exit()

    json_path = str(sys.argv[1])
    run_default(json_path, dict(silent=False))

