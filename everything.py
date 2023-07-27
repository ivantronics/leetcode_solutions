import collections
from collections import deque, Counter, defaultdict
from typing import List, Optional
import math
import heapq


# Helper classes
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class GraphNode:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TrieNode:

    def __init__(self):
        self.children = {}
        self.endofword = False

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.isWord = False
            self.refs = 0

    def addWord(self, word):
        cur = self
        cur.refs += 1
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.refs += 1
        cur.isWord = True

    def removeWord(self, word):
        cur = self
        cur.refs -= 1
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
                cur.refs -= 1


class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class UnionFind:
    def __init__(self):
        self.f = {}

    def findParent(self, x):
        y = self.f.get(x, x)
        if x != y:
            y = self.f[x] = self.findParent(y)
        return y

    def union(self, x, y):
        self.f[self.findParent(x)] = self.findParent(y)


# Solutions in order
class Solution:
    # 1. Two Sum
    def twoSum(self, nums: list[int], target: int):  # -> list[int]:
        lut: dict = {}
        for idx, val in enumerate(nums):
            needed = target - val
            if needed in lut:
                return [lut[needed], idx]
            lut[val] = idx
        return [0, 0]

    # 2. Add Two Numbers
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode
        curr = dummy
        carry = 0

        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0

            val = v1 + v2 + carry

            carry = val // 10
            val = val % 10

            curr.next = ListNode(val)

            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy.next

    # 3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        currstring = set()
        l = 0
        longest = 0
        for r in range(len(s)):
            while s[r] in currstring:
                currstring.remove(s[l])
                l += 1
            currstring.add(s[r])
            longest = max(longest, r - l + 1)
        return longest

    # 4. Median of Two Sorted Arrays
    def findMedianSortedArrays(self, nums1: list[int],
                               nums2: list[int]) -> float:
        A, B = nums1, nums2
        total = len(A) + len(B)
        half = total // 2
        if len(A) > len(B):
            A, B = B, A
        left, right = 0, len(A) - 1
        while True:
            i = (left + right) // 2  # A
            j = half - i - 2  # B

            Aleft = A[i] if i >= 0 else float("-infinity")
            Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")

            Bleft = B[j] if j >= 0 else float("-infinity")
            Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

            if Aleft <= Bright and Bleft <= Aright:
                if total % 2:
                    return min(Aright, Bright)
                else:
                    return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                right = i - 1
            else:
                left = i + 1

    # 5. Longest Palindromic Substring
    def longestPalindrome(self, s: str) -> str:
        res = ""
        resLen = 0

        for i in range(len(s)):
            l = r = i
            while l >= 0 and r < len(s) and s[r] == s[l]:
                curr = r - l + 1
                if curr > resLen:
                    res = s[l: r + 1]
                    resLen = curr
                l -= 1
                r += 1
            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[r] == s[l]:
                curr = r - l + 1
                if curr > resLen:
                    res = s[l: r + 1]
                    resLen = curr
                l -= 1
                r += 1
        return res

    # 7. Reverse Integer
    def reverse(self, x: int) -> int:
        MIN = -2147483648  # -2^31,
        MAX = 2147483647  #  2^31 - 1

        res = 0
        while x:
            digit = int(math.fmod(x, 10))  # (python dumb) -1 %  10 = 9
            x = int(x / 10)  # (python dumb) -1 // 10 = -1

            if res > MAX // 10 or (res == MAX // 10 and digit > MAX % 10):
                return 0
            if res < MIN // 10 or (res == MIN // 10 and digit < MIN % 10):
                return 0
            res = (res * 10) + digit

        return res

    def reverse(self, x: int) -> int:
        if x == 0:
            return x
        par = x // abs(x)
        x = abs(x)
        res = 0

        while x:
            res = res*10+(x%10)
            x = x//10
        res *= par
        return res if -2**31<res<2**31-1 else 0

    # 10. Regular Expression Matching
    def isMatch(self, s: str, p: str) -> bool:
        cache = {}

        def dfs(i, j):
            if (i, j) in cache:
                return cache[(i, j)]
            if i >= len(s) and j >= len(p):
                return True
            if j >= len(p):
                return False

            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if (j + 1) < len(p) and p[j + 1] == "*":
                cache[(i, j)] = dfs(i, j + 2) or (  # dont use *
                        match and dfs(i + 1, j)
                )  # use *
                return cache[(i, j)]
            if match:
                cache[(i, j)] = dfs(i + 1, j + 1)
                return cache[(i, j)]
            cache[(i, j)] = False
            return False

        return dfs(0, 0)

    # 11. Container with the most water

    def maxArea(self, height: list[int]) -> int:  # BRUTE FORCE
        result: int = 0
        for idx1, left in enumerate(height):
            for idx2, right in enumerate(height[idx1 + 1:]):
                result = max(result, min(left, right) * (idx2 + 1))
        return result

    def maxArea2(self, height: list[int]) -> int:  # LINEAR two-pointer
        result: int = 0
        left, right = 0, len(height) - 1
        while left < right:
            result = max(result,
                         min(height[left], height[right]) * (right - left))
            if height[left] <= height[right]:
                left += 1
            elif height[left] > height[right]:
                right -= 1
        return result

    # 15 3Sum.
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        res = []

        for idx, val in enumerate(nums):
            if idx > 0 and val == nums[idx - 1]:
                continue
            left, right = idx + 1, len(nums) - 1
            while left < right:
                check_sum = val + nums[left] + nums[right]
                if check_sum < 0:
                    left += 1
                elif check_sum > 0:
                    right -= 1
                else:
                    res.append([val, nums[left], nums[right]])
                    left += 1
                    while nums[left] == nums[left - 1] and left < right:
                        left += 1
        return res

    # 17. Letter Combinations of a Phone Number
    def letterCombinations(self, digits: str) -> List[str]:
        res = []
        digitToChar = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "qprs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(i, currstr):
            if len(currstr) == len(digits):
                res.append(currstr)
                return
            for c in digitToChar[digits[i]]:
                backtrack(i + 1, currstr + c)

        if digits:
            backtrack(0, "")

        return res

    # 19. Remove Nth Node From End of List
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        left = dummy
        right = head

        while n > 0 and right:
            right = right.next
            n -= 1

        while right:
            left = left.next
            right = right.next

        # removing the node
        left.next = left.next.next

        return dummy.next

    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        stack = []
        hashmap = {")": "(", "]": "[", "}": "{"}
        for char in s:
            if stack and stack[-1] == hashmap.get(char):
                stack.pop()
            else:
                stack.append(char)
        return not stack

    # 21. Merge Two Sorted Lists
    def mergeTwoLists(self, list1: Optional[ListNode],
                      list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        tail = dummy

        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next

        if list1:
            tail.next = list1
        elif list2:
            tail.next = list2

        return dummy

    # 22. Generate Parentheses
    def generateParenthesis(self, n: int) -> list[str]:
        res = []
        stack = []

        def backtrack(openN, closedN):
            if openN == closedN == n:
                res.append("".join(stack))
                return
            if openN < n:
                stack.append("(")
                backtrack(openN + 1, closedN)
                stack.pop()
            if openN > closedN:
                stack.append(")")
                backtrack(openN, closedN + 1)
                stack.pop()

        backtrack(0, 0)
        return res

    # 23. Merge k Sorted Lists
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists or len(lists) == 0:
            return None
        while len(lists) > 1:
            mergedlist = []

            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if (i + 1) < len(lists) else None
                mergedlist.append(self.combinelists(l1, l2))

            lists = mergedlist
        return lists[0]

    def combinelists(self, l1, l2):
        dummy = ListNode()
        tail = dummy

        while l1 and l2:
            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next
        tail.next = l1 if l1 else l2
        return dummy.next

    # 25. Reverse Nodes in k-Group
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        previousgroup = dummy

        while True:
            kth = self.gotok(previousgroup, k)
            if not kth:
                break
            nextgroup = kth.next

            prev, curr = kth.next, previousgroup.next

            while curr != nextgroup:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp

            tmp = previousgroup.next
            previousgroup.next = kth
            previousgroup = tmp

        return dummy.next

        def gotok(self, curr, k):
            while curr and k > 0:
                curr = curr.next
                k -= 1
            return curr

    # 33. Clone Graph (i have no idea why there are two 33s)
    def cloneGraph(self, node: 'GraphNode') -> 'GraphNode':
        old_to_new = {}

        def dfs(node):
            if node in old_to_new:
                return old_to_new[node]

            copy = GraphNode(node.val)
            old_to_new[node] = copy
            for neighbor in node.neighbors:
                copy.neighbors.append(dfs(neighbor))

            return copy

        return dfs(node) if node else None

    # 33. Search in Rotated Sorted Array
    def searchra(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[left]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if target < nums[mid] or target > nums[right]:
                    right = mid - 1
                else:
                    left = mid + 1
        return -1

    # 36. Valid Sudoku
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        sqrs = collections.defaultdict(set)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r] or
                        board[r][c] in cols[c] or
                        board[r][c] in sqrs[(r // 3, c // 3)]):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                sqrs[(r // 3, c // 3)].add(board[r][c])

        return True

    # 39. Combination Sum
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(i, curr, total):
            if total == target:
                res.append(curr.copy())
                return
            if i >= len(candidates) or total > target:
                return

            curr.append(candidates[i])
            dfs(i, curr, total + candidates[i])
            curr.pop()
            dfs(i + 1, curr, total)

        dfs(0, [], 0)
        return res

    # 40. Combination Sum II
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()

        def backtrack(curr, pos, target):
            if target == 0:
                res.append(curr[:])
                return
            if target <= 0:
                return
            prev = -1
            for i in range(pos, len(candidates)):
                if candidates[i] == prev:
                    continue
                curr.append(candidates[i])
                backtrack(curr, i + 1, target - candidates[i])
                curr.pop()
                prev = candidates[i]

        backtrack([], 0, target)

        return res

    # 42. Trapping rain water.
    def trap(self, height: list[int]) -> int:  # Linear space, linear time
        result = 0
        for idx, val in enumerate(height):
            around = min(max(height[:idx], default=0), max(height[idx:]))
            checksquare = around - val
            if checksquare > 0:
                result += checksquare
            else:
                continue
        return result

    def trap2(self, height: list[int]) -> int:  # Linear with less memory
        result = 0
        if not height: return result
        left, right = 0, len(height) - 1
        maxleft, maxright = height[left], height[right]
        while left < right:
            if maxleft <= maxright:
                left += 1
                maxleft = max(maxleft, height[left])
                result += maxleft - height[left]
            else:
                right -= 1
                maxright = max(maxright, height[right])
                result += maxright - height[right]
        return result

    # 43. Multiply Strings
    def multiply(self, num1: str, num2: str) -> str:
        if "0" in [num1, num2]:
            return "0"
        res = [0] * (len(num1) + len(num2))
        num1, num2 = num1[::-1], num2[::-1]

        for i1 in range(len(num1)):
            for i2 in range(len(num2)):
                digit = int(num1[i1]) * int(num2[i2])
                res[i1 + i2] += digit
                res[i1 + i2 + 1] += res[i1 + i2] // 10
                res[i1 + i2] = res[i1 + i2] % 10

        res, beg = res[::-1], 0
        while beg < len(res) and res[beg] == 0:
            beg += 1

        res = map(str, res[beg:])

        return "".join(res)

    #45. Jump Game II
    def jump(self, nums: List[int]) -> int:
        res = 0
        l = r = 0

        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            res += 1

        return res

    # 46. Permutations
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        if len(nums) == 1:
            return [nums[:]]

        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)

            for perm in perms:
                perm.append(n)
            res.extend(perms)
            nums.append(n)

        return res

    # 48. Rotate Image
    def rotate(self, matrix: List[List[int]]) -> None:
        l, r = 0, len(matrix) - 1
        while l < r:
            for i in range(r - l):
                top, bottom = l, r

                matrix[top][l + i], matrix[bottom - i][l], matrix[bottom][r - i], matrix[top + i][r] = matrix[bottom - i][l], matrix[bottom][r - i], matrix[top + i][r], matrix[top][l + i]
            l += 1
            r -= 1

    # 49. Group Anagrams
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        result_table: dict = {}

        for string in strs:
            word_map = [0] * 26
            for c in string:
                word_map[ord(c) - ord("a")] += 1
            if tuple(word_map) in result_table:
                result_table[tuple(word_map)].append(string)
            else:
                result_table[tuple(word_map)] = [string]
        return list(result_table.values())

    def groupAnagrams2(self, strs: list[
        str]):  # -> list[list[str]]:  # Faster but with sorting
        results = {}
        for word in strs:
            key = "".join(sorted(word))
            if key in results:
                results[key].append(word)
            else:
                results[key] = [word]
        return results.values()

    # 50. Pow(x, n)
    def myPow(self, x: float, n: int) -> float:

        def helper(x, n):
            if x == 0:
                return 0
            if n == 0:
                return 1
            res = helper(x, n // 2)
            res *= res
            return x * res if n % 2 else res

        res = helper(x, abs(n))

        return res if n >= 0 else 1/res

    # 51. N-Queens
    def solveNQueens(self, n: int) -> List[List[str]]:
        col = set()
        posdiag = set() # (r + c)
        negdiag = set() # (r - c)

        res = []
        board = [["."] * n for _ in range(n)]

        def backtrack(r):
            if r == n:
                res.append(["".join(row) for row in board])
                return
            for c in range(n):
                if c in col or (r + c) in posdiag or (r - c) in negdiag:
                    continue
                col.add(c)
                posdiag.add(r + c)
                negdiag.add(r - c)
                board[r][c] = "Q"
                backtrack(r + 1)
                col.remove(c)
                posdiag.remove(r + c)
                negdiag.remove(r - c)
                board[r][c] = "."
        backtrack(0)
        return res

    # 53. Maximum Subarray
    def maxSubArray(self, nums: List[int]) -> int:
        maxsub = nums[0]
        currsum = 0

        for n in nums:
            if currsum < 0:
                currsum = 0
            currsum += n
            maxsub = max(maxsub, currsum)

        return maxsub

    # 54. Spiral Matrix
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)
        res = []

        while left < right and top < bottom:
            for i in range(left, right):
                res.append(matrix[top][i])
            top += 1

            for i in range(top, bottom):
                res.append(matrix[i][right - 1])
            right -= 1

            if not (left < right and top < bottom):
                break

            for i in range(right - 1, left - 1, -1):
                res.append(matrix[bottom - 1][i])
            bottom -= 1

            for i in range(bottom - 1, top - 1, -1):
                res.append(matrix[i][left])
            left += 1

        return res

    # 55. Jump Game
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) - 1

        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= goal:
                goal = i

        return goal == 0

    # 56. Merge Intervals
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        output = [intervals[0]]

        for start, end in intervals[1:]:
            lastEnd = output[-1][-1]
            if start <= lastEnd:
                output[-1][-1] = max(lastEnd, end)
            else:
                output.append([start, end])

        return output

    # 57. Insert Interval
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []

        for i in range(len(intervals)):
            if newInterval[-1] < intervals[i][0]:
                res.append(newInterval)
                return res + intervals[i:]
            elif newInterval[0] > intervals[i][-1]:
                res.append(intervals[i])
            else:
                newInterval = [min(newInterval[0], intervals[i][0]), max(newInterval[-1], intervals[i][-1])]

        res.append(newInterval)

        return res

    # 62. Unique Paths
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * n
        for i in range(m - 1):
            newRow = [1] * n
            for j in range(n - 2, -1, -1):
                newRow[j] = newRow[j + 1] + row[j]
            row = newRow
        return row[0]

    # 66. Plus One
    def plusOne(self, digits: List[int]) -> List[int]:
        digits = digits[::-1]
        carry, i = 1, 0

        while carry:
            if i < len(digits):
                if digits[i] == 9:
                    digits[i] = 0
                else:
                    digits[i] += 1
                    carry = 0
            else:
                digits.append(1)
                carry = 0
            i += 1
        return digits[::-1]

    # 70. Climbing Stairs
    def climbStairs(self, n: int) -> int:
        one = two = 1

        for _ in range(n - 1):
            one, two = one + two, one

        return one

    # 72. Edit Distance
    def minDistance(self, word1: str, word2: str) -> int:
        cache = [[float("inf")] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for j in range(len(word2) + 1):
            cache[len(word1)][j] = len(word2) - j
        for i in range(len(word1) + 1):
            cache[i][len(word2)] = len(word1) - i

        for i in range(len(word1) - 1, -1, -1):
            for j in range(len(word2) -1, -1, -1):
                if word1[i] == word2[j]:
                    cache[i][j] = cache[i + 1][j + 1]
                else:
                    cache[i][j] = 1 + min(cache[i + 1][j], cache[i][j + 1], cache[i + 1][j + 1])

        return cache[0][0]

    # 73. Set Matrix Zeroes
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        ROWS, COLS = len(matrix), len(matrix[0])
        rowZero = False

        for r in range(ROWS):
            for c in range(COLS):
                if matrix[r][c] == 0:
                    matrix[0][c] = 0
                    if r > 0:
                        matrix[r][0] = 0
                    else:
                        rowZero = True

        for r in range(1, ROWS):
            for c in range(1, COLS):
                if matrix[0][c] == 0 or matrix[r][0] == 0:
                    matrix[r][c] = 0

        if matrix[0][0] == 0:
            for r in range(ROWS):
                matrix[r][0] = 0

        if rowZero:
            for c in range(COLS):
                matrix[0][c] = 0

    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: list[list[int]], target: int) -> bool:
        first, last = 0, len(matrix) - 1
        while first <= last:
            matrix_mid = (first + last) // 2
            if matrix[matrix_mid][-1] > target and matrix[matrix_mid][
                0] > target:
                last = matrix_mid - 1
            elif matrix[matrix_mid][-1] < target and matrix[matrix_mid][
                0] < target:
                first = matrix_mid + 1
            else:
                left, right = 0, len(matrix[matrix_mid]) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if matrix[matrix_mid][mid] < target:
                        left = mid + 1
                    elif matrix[matrix_mid][mid] > target:
                        right = mid - 1
                    else:
                        return True
                return False
        return False

    # 76. Minimum Window Substring
    def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""

        countT, countW = {}, {}

        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, reslen = [-1, 1], float("infinity")
        l = 0
        for r in range(len(s)):
            c = s[r]
            countW[c] = 1 + countW.get(c, 0)

            if c in countT and countW[c] == countT[c]:
                have += 1

            while have == need:
                if (r - l + 1) < reslen:
                    res = [l, r]
                    reslen = r - l + 1
                countW[s[l]] -= 1
                if s[l] in countT and countW[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1
        l, r = res
        return s[l:r + 1] if reslen != float("infinity") else ""

    # 78. Subsets
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []

        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            dfs(i + 1)

        dfs(0)

        return res

    # 79. Word Search
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        path = set()
        if len(word) > ROWS*COLS: return False
        # [] is the starting value in sum, flattening 2d board to 1d list
        count = Counter(sum(board, []))
        for c, countWord in Counter(word).items():
            if count[c] < countWord:
                return False
        # optimization to reduce dfs time
        if count[word[0]] > count[word[-1]]:
            word = word[::-1]

        def dfs(r, c, i):
            if i == len(word):
                return True
            if (r < 0 or c < 0) or \
                    (r >= ROWS or c >= COLS) or \
                    (word[i] != board[r][c]) or \
                    ((r, c) in path):
                return False
            path.add((r, c))
            res = (dfs(r + 1, c, i + 1) or
                   dfs(r, c + 1, i + 1) or
                   dfs(r - 1, c, i + 1) or
                   dfs(r, c - 1, i + 1))
            path.remove((r, c))
            return res

        for r in range(ROWS):
            for c in range(COLS):
                if dfs(r, c, 0):
                    return True

        return False

    # 84. Largest Rectangle in Histogram
    def largestRectangleArea(self, heights: list[int]):  # -> int:
        stack = [(0, heights[0])]
        length = len(heights)
        maxarea = 0
        newindex = None
        for idx, val in enumerate(heights):
            if val >= stack[-1][-1]:
                stack.append((idx, val))  # type: ignore 
            else:
                while stack and val < stack[-1][-1]:
                    currarea = (idx - stack[-1][0]) * stack[-1][1]
                    maxarea = max(maxarea, currarea)
                    newindex = stack[-1][0]
                    stack.pop()
                stack.append((newindex, val))  # type: ignore 
        for bar in stack:
            currarea = (length - bar[0]) * bar[1]
            maxarea = max(maxarea, currarea)
        return maxarea

    # 90. Subsets II
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()

        def backtrack(i, subset):
            if i == len(nums):
                res.append(subset[:])
                return
            subset.append(nums[i])
            backtrack(i + 1, subset)
            subset.pop()
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            backtrack(i + 1, subset)

        backtrack(0, [])

        return res

    # 91. Decode Ways
    def numDecodings(self, s: str) -> int:
        dp = {len(s): 1}

        def dfs(i):
            if i in dp:
                return dp[i]
            if s[i] == "0":
                return 0
            res = dfs(i + 1)
            if i + 1 < len(s) and (s[i] == "1" or s[i] == "2" and s[i + 1] in "0123456"):
                res += dfs(i + 2)
            dp[i] = res

        return dfs(0)

    # dp
    def numDecodings(self, s: str) -> int:
        dp = {len(s): 1}
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "0":
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]

            if i + 1 < len(s) and (
                    s[i] == "1" or s[i] == "2"
                    and s[i + 1] in "0123456"):
                dp[i] += dp[i + 2]
        return dp[0]

    # 97. Interleaving String
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [[False] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        dp[len(s1)][len(s2)] = True

        for i in range(len(s1), -1, -1):
            for j in range(len(s2), -1, -1):
                if i < len(s1) and s1[i] == s3[i + j] and dp[i + 1][j]:
                    dp[i][j] = True
                if j < len(s2) and s2[j] == s3[i + j] and dp[i][j + 1]:
                    dp[i][j] = True

        return dp[0][0]

    # 98. Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node, left, right):
            if not node:
                return True
            if not (left < node.val < right):
                return False
            return validate(node.left, left, node.val) and validate(node.right, node.val, right)
        return validate(root, float("-inf"), float("+inf"))

    # 100. Same Tree
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False

        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # 102. Binary Tree Level Order Traversal
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = []
        level = []
        q = deque()
        q.append(root)

        while q:
            qLen = len(q)
            for i in range(qLen):
                node = q.popleft()
                if node:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
            if level:
                res.append(level)
            level = []
        return res

    # 104. Maximum Depth of Binary Tree
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        level = 1
        q = deque([root])

        while q:
            for i in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(q.left)
                elif q.right:
                    q.append(q.right)
            level += 1

        return level

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        stack = [[root, 1]]
        res = 0
        while stack:
            node, depth = stack.pop()

            if node:
                res = max(res, depth)
                stack.append([node.left, depth + 1])
                stack.append([node.right, depth + 1])

        return res

    # 105. Construct Binary Tree from Preorder and Inorder Traversal
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])

        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])

        return root

    # 110. Balanced Binary Tree
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def dfs(root):
            if not root: return [True, 0]

            left, right = dfs(root.left), dfs(root.right)

            balanced = (abs(left[1] - right[1]) <= 1) and left[0] and right[0]

            return [balanced, 1 + max(left[1], right[1])]

        return dfs(root)[0]

    # 115. Distinct Subsequences
    def numDistinct(self, s: str, t: str) -> int:
        cache = {}

        def dfs(i, j):
            if j == len(t):
                return 1
            if i == len(s):
                return 0
            if (i, j) in cache:
                return cache[(i, j)]

            if s[i] == t[j]:
                cache[(i, j)] = dfs(i + 1, j + 1) + dfs(i + 1, j)
            else:
                cache[(i, j)] = dfs(i + 1, j)

            return cache[(i, j)]

        return dfs(0, 0)

    # 121. Best Time to Buy and Sell Stock
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1
        max_profit = 0

        while r < len(prices):
            if prices[l] < prices[r]:
                profit = prices[r] - prices[l]
                max_profit = max(max_profit, profit)
            else:
                l = r
            r += 1

        return max_profit

    # 124. Binary Tree Maximum Path Sum
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = [root.val]

        def dfs(node):
            if not node:
                return 0
            leftmax = max(dfs(node.left), 0)
            rightmax = max(dfs(node.right), 0)

            res[0] = max(res[0], node.val + leftmax + rightmax)

            return node.val + max(leftmax, rightmax)

        dfs(root)
        return res[0]

    # 125. Valid Palindrome.
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s)
        while l < r:
            while l < r and not self.isAlnum(s[l]):
                l += 1
            while r > l and not self.isAlnum(s[r]):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True

    def isAlnum(self, c: str) -> bool:
        return ord("A") <= ord(c) <= ord("Z") or \
            ord('a') <= ord(c) <= ord("z") or \
            ord('0') <= ord(c) <= ord("9")

    def isPalindrome2(self, s: str) -> bool:  # Faster with built-in methods
        s = "".join([c.lower() for c in s if c.isalnum()])
        return s == s[::-1]

    # 127. Word Ladder
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        neighbors = defaultdict(list)
        wordList.append(beginWord)
        visited = set([beginWord])
        q = deque([beginWord])
        res = 1
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + "*" + word[i + 1:]
                neighbors[pattern].append(word)
        while q:
            for i in range(len(q)):
                word = q.popleft()
                if word == endWord:
                    return res
                for j in range(len(word)):
                    pattern = word[:j] + "*" + word[j + 1:]
                    for nei in neighbors[pattern]:
                        if nei not in visited:
                            visited.add(nei)
                            q.append(nei)
            res += 1
        return 0

    # 128. Longest Consecutive Sequence
    def longestConsecutive(self, nums: list[int]) -> int:
        hashmap = set(nums)
        longest = 0
        for n in nums:
            if (n - 1) not in hashmap:
                curr_length = 0
                while n + curr_length in hashmap:
                    curr_length += 1
                longest = max(curr_length, longest)
        return longest

    # 130. Surrounded Regions
    def solve(self, board: List[List[str]]) -> None:
        ROWS, COLS = len(board), len(board[0])

        def capture(r, c):
            if r < 0 or c < 0 \
                    or r == ROWS \
                    or c == COLS \
                    or board[r][c] != "O":
                return

            board[r][c] = "T"
            capture(r + 1, c)
            capture(r - 1, c)
            capture(r, c + 1)
            capture(r, c - 1)

        for r in range(ROWS):
            for c in range(COLS):
                if (board[r][c] == "O" and (r in [0, ROWS - 1] or c in [0, COLS - 1])):
                    capture(r, c)

        for r in range(ROWS):
            for c in range(COLS):
                if board[r][c] == "O":
                    board[r][c] = "X"
                if board[r][c] == "T":
                    board[r][c] = "O"

    # 131. Palindrome Partitioning
    def partition(self, s: str) -> List[List[str]]:
        res = []
        sub = []

        def dfs(i):
            if i >= len(s):
                res.append(sub[:])
                return
            for j in range(i, len(s)):
                if self.ispalindrome(s, i, j):
                    sub.append(s[i:j+1])
                    dfs(j + 1)
                    sub.pop()
        dfs(0)
        return res

    def ispalindrome(self, s, l, r):
        while l < r:
            if s[l] != s[r]:
                return False
            l, r = l + 1, r - 1
        return True

    # 134. Gas Station
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        total = 0
        res = 0

        for i in range(len(gas)):
            total += (gas[i] - cost[i])
            if total < 0:
                total = 0
                res = i + 1

        return res

    # 136. Single Number
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for n in nums:
            res = n ^ res
        return res

    # 138. Copy List with Random Pointer
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        oldvals = {None: None}
        curr = head
        while curr:
            copy = Node(curr.val)
            oldvals[curr] = copy
            curr = curr.next
        curr = head
        while curr:
            copy = oldvals[curr]
            copy.next = oldvals[curr.next]
            copy.random = oldvals[curr.random]
            curr = curr.next

        return oldvals[head]

    # 139. Word Break
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1)
        dp[len(s)] = True

        for i in range(len(s) - 1, -1, -1):
            for w in wordDict:
                if (i + len(w) <= len(s) and s[i:i+len(w)] == w):
                    dp[i] = dp[i + len(w)]
                if dp[i]:
                    break

        return dp[0]

    # 141. Linked List Cycle
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    # 143. Reorder List
    def reorderList(self, head: Optional[ListNode]) -> None:
        slow, fast = head, head.next

        while fast and fast.next:  # finding middle
            slow = slow.next
            fast = fast.next.next

        secondhalf = slow.next
        slow.next = None
        previous = None

        while secondhalf:
            temp = secondhalf.next
            secondhalf.next = previous
            previous = secondhalf
            secondhalf = temp

        # merging two halves
        firsthalf, secondhalf = head, previous
        while secondhalf:
            temp1, temp2 = firsthalf.next, secondhalf.next
            firsthalf.next = secondhalf
            secondhalf.next = temp1
            firsthalf, secondhalf = temp1, temp2

    # 146. LRU Cache
    class LRUCache:

        def __init__(self, capacity: int):
            self.capacity = capacity
            self.hashmap = {}  # maps key to node
            self.left, self.right = Node(0, 0), Node(0, 0)
            self.left.next, self.right.prev = self.right, self.left

        def remove(self, node):
            prev, nxt = node.prev, node.next
            prev.next, nxt.prev = nxt, prev

        def insert(self, node):
            prev, nxt = self.right.prev, self.right
            prev.next = nxt.prev = node
            node.next, node.prev = nxt, prev

        def get(self, key: int) -> int:
            if key in self.hashmap:
                self.remove(self.hashmap[key])
                self.insert(self.hashmap[key])
                return self.hashmap[key].val
            return -1

        def put(self, key: int, value: int) -> None:
            if key in self.hashmap:
                self.remove(self.hashmap[key])
            self.hashmap[key] = Node(key, value)
            self.insert(self.hashmap[key])

            if len(self.hashmap) > self.capacity:
                lru = self.left.next
                self.remove(lru)
                del self.hashmap[lru.key]

    # 150. Evaluate Reverse Polish Notation
    def evalRPN(self, tokens: list[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                stack.append(-stack.pop() + stack.pop())
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                stack.append(int(1 / stack.pop() * stack.pop()))
            else:
                stack.append(int(c))
        return stack[-1]

    # 152. Maximum Product Subarray
    def maxProduct(self, nums: List[int]) -> int:
        res = max(nums)
        curmin = curmax = 1

        for n in nums:
            curmax, curmin = max(n * curmax, n * curmin, n), min(curmax * n, n * curmin, n)
            res = max(res, curmax)
        return res

    # 153. Find Minimum in Rotated Sorted Array
    def findMin(self, nums: list[int]) -> int:
        left, right = 0, len(nums) - 1
        res = nums[left]
        while left <= right:
            if nums[left] < nums[right]:
                res = min(nums[left], res)
                break
            mid = (left + right) // 2
            res = min(res, nums[mid])
            if nums[mid] >= nums[left]:
                left = mid + 1
            else:
                right = mid - 1
        return res

    # 155. Min Stack
    class MinStack:

        def __init__(self):
            self.stack = []
            self.minstack = []

        def push(self, val: int) -> None:
            self.stack.append(val)
            minval = min(self.minstack[-1] if self.minstack else val, val)
            self.minstack.append(minval)

        def pop(self) -> None:
            self.stack.pop()
            self.minstack.pop()

        def top(self) -> int | None:
            if self.stack:
                return self.stack[-1]
            else:
                return None

        def getMin(self) -> int | None:
            if self.minstack:
                return self.minstack[-1]
            else:
                return None

    # 167 Two Sum II. Input array is sorted
    def twoSumII(self, numbers: list[int], target: int) -> list[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            checksum = numbers[left] + numbers[right]
            if checksum == target:
                return [left + 1, right + 1]
            elif checksum > target:
                right -= 1
            elif checksum < target:
                left += 1
        return [-1, -1]

    # 169. Majority Element
    def majorityElement(self, nums: List[int]) -> int:
        res = count = 0

        for n in nums:
            if count == 0:
                res = n
            count += (1 if n == res else -1)

        return res

    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]

    # 178. Graph Valid Tree
    def valid_tree(self, n: int, edges: List[List[int]]) -> bool:
        if not n:
            return True

        adjlist = {i:[] for i in range(n)}
        visited = set()

        for n1, n2 in edges:
            adjlist[n1].append(n2)
            adjlist[n2].append(n1)

        def dfs(node, prev):
            if node in visited:
                return False

            visited.add(node)

            for i in adjlist[node]:
                if i == prev:
                    continue
                if not dfs(i, node):
                    return False

            return True

        return dfs(0, -1) and len(visited) == n

    # Alternative

    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def __find(self, n: int) -> int:
        while n != self.parents.get(n, n):
            n = self.parents.get(n, n)
        return n

    def __connect(self, n: int, m: int) -> None:
        pn = self.__find(n)
        pm = self.__find(m)
        if pn == pm:
            return
        if self.heights.get(pn, 1) > self.heights.get(pm, 1):
            self.parents[pn] = pm
        else:
            self.parents[pm] = pn
            self.heights[pm] = self.heights.get(pn, 1) + 1
        self.components -= 1

    def valid_tree(self, n: int, edges: List[List[int]]) -> bool:
        # init here as not sure that ctor will be re-invoked in different tests
        self.parents = {}
        self.heights = {}
        self.components = n

        for e1, e2 in edges:
            if self.__find(e1) == self.__find(e2):  # 'redundant' edge
                return False
            self.__connect(e1, e2)

        return self.components == 1  # forest contains one tree

    # 190. Reverse Bits
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            bit = (n >> i) & 1
            res = res | (bit << (31 - i))
        return res

    # 191. Number of 1 Bits
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += n % 2
            n = n >> 1
        return res

    # 198. House Robber
    def rob(self, nums: List[int]) -> int:
        a = b = 0
        for i in nums:
            a, b = b, max(i + a, b)
        return b

    # 199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        q = deque([root])

        while q:
            rightside = None
            qLen = len(q)

            for i in range(qLen):
                node = q.popleft()
                if node:
                    rightside = node
                    q.append(node.left)
                    q.append(node.right)

            if rightside:
                res.append(rightside.val)

        return res

    # 200. Number of Islands
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        visited = set()
        islands = 0

        def dfs(r, c):
            if (r not in range(rows)
                    or c not in range(cols)
                    or grid[r][c] == "0"
                    or (r, c) in visited
            ):
                return

            visited.add((r, c))
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

            for dr, dc in directions:
                dfs(r + dr, c + dc)

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visited:
                    islands += 1
                    dfs(r, c)
        return islands

    # 202. Happy Number
    def isHappy(self, n: int) -> bool:
        visited = set()

        while n not in visited:
            visited.add(n)
            n = self.sumOfSquares(n)

            if n == 1:
                return True

        return False

    def sumOfSquares(self, n):
        output = 0

        while n:
            digit = n % 10
            digit = digit ** 2
            output += digit
            n = n // 10

        return output

    # 206. Reverse Linked List
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head

        while curr:
            tempval = curr.next  # temporary value so not to lose it when reassigning
            curr.next = prev
            prev = curr
            curr = tempval

        return prev

    # 207. Course Schedule
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        prereqMap = {i:[] for i in range(numCourses)}
        visited = set()
        for crs, pre in prerequisites:
            prereqMap[crs].append(pre)

        def dfs(crs):
            if crs in visited:
                return False
            if not prereqMap[crs]:
                return True
            visited.add(crs)
            for pre in prereqMap[crs]:
                if not dfs(pre): return False
            visited.remove(crs)
            prereqMap[crs] = []
            return True

        for crs in range(numCourses):
            if not dfs(crs): return False

        return True

    # 208. Implement Trie (Prefix Tree)
    class Trie:

        def __init__(self):
            self.root = TrieNode()

        def insert(self, word: str) -> None:
            curr = self.root
            for c in word:
                if c not in curr.children:
                    curr.children[c] = TrieNode()
                curr = curr.children[c]

            curr.endofword = True

        def search(self, word: str) -> bool:
            curr = self.root

            for c in word:
                if c not in curr.children:
                    return False
                curr = curr.children[c]

            return curr.endofword

        def startsWith(self, prefix: str) -> bool:
            curr = self.root

            for c in prefix:
                if c not in curr.children:
                    return False
                curr = curr.children[c]
            return True

    # 210. Course Schedule II
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        pMap = {c:[] for c in range(numCourses)}
        visit, done = set(), set()
        output = []
        for crs, pre in prerequisites:
            pMap[crs].append(pre)

        def dfs(crs):
            if crs in done:
                return False
            if crs in visit:
                return True

            done.add(crs)

            for pre in pMap[crs]:
                if not dfs(pre):
                    return False

            done.remove(crs)
            visit.add(crs)
            output.append(crs)

            return True

        for c in range(numCourses):
            if not dfs(c):
                return []

        return output

    # 211. Design Add and Search Words Data Structure
    class WordDictionary:

        def __init__(self):
            self.root = TrieNode()

        def addWord(self, word: str) -> None:
            curr = self.root

            for c in word:
                if c not in curr.children:
                    curr.children[c] = TrieNode()
                curr = curr.children[c]

            curr.endofword = True

        def search(self, word: str) -> bool:

            def dfs(j, node):
                curr = node

                for i in range(j, len(word)):
                    c = word[i]

                    if c == ".":
                        for candidate in curr.children.values():
                            if dfs(i + 1, candidate):
                                return True
                        return False
                    else:
                        if c not in curr.children:
                            return False
                        curr = curr.children[c]

                return curr.endofword

            return dfs(0, self.root)

    # 212. Word Search II
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addWord(w)

        ROWS, COLS = len(board), len(board[0])
        res, visit = set(), set()

        def dfs(r, c, node, word):
            if (
                    r not in range(ROWS)
                    or c not in range(COLS)
                    or board[r][c] not in node.children
                    or node.children[board[r][c]].refs < 1
                    or (r, c) in visit
            ):
                return

            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.isWord:
                node.isWord = False
                res.add(word)
                root.removeWord(word)

            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visit.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")

        return list(res)

    # 213. House Robber II
    def rob(self, nums: List[int]) -> int:
        return max(self.robber(nums[:-1]), self.robber(nums[1:]), nums[0])

    def robber(self, nums: List[int]) -> int:
        a = b = 0
        for i in nums:
            a, b = b, max(i + a, b)
        return b

    # 215. Kth Largest Element in an Array
    def findKthLargest(self, nums: List[int], k: int) -> int:
        k = len(nums) - k

        def quickselect(l, r):
            pivot, p = nums[r], l
            for i in range(l, r):
                if nums[i] <= pivot:
                    nums[p], nums[i] = nums[i], nums[p]
                    p += 1
            nums[p], nums[r] = nums[r], nums[p]

            if k < p: return quickselect(l, p - 1)
            elif k > p: return quickselect(p + 1, r)
            else: return nums[p]

        return quickselect(0, len(nums) - 1)

    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums)-k]

    # 226. Invert Binary Tree
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        root.left, root.right = root.right, root.left

        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        curr = root

        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left

            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val

            curr = curr.right

    # 235. Lowest Common Ancestor of a Binary Search Tree
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        curr = root

        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr

    # 238.Product of Array Except Self
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        result = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            result[i] = prefix
            prefix *= nums[i]

        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            result[i] *= postfix
            postfix *= nums[i]

        return result

    # 239. Sliding Window Maximum
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q = collections.deque()
        l = r = 0

        while r < len(nums):
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            if l > q[0]:
                q.popleft()

            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1

        return output

    # 268. Missing Number
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)

        for i in range(len(nums)):
            res += (i - nums[i])

        return res

    # 269. Alien Dictionary
    def alienOrder(self, words: List[str]) -> str:
        adj = {char: set() for word in words for char in word}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            minLen = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:minLen] == w2[:minLen]:
                return ""
            for j in range(minLen):
                if w1[j] != w2[j]:
                    print(w1[j], w2[j])
                    adj[w1[j]].add(w2[j])
                    break

        visited = {}  # {char: bool} False visited, True current path
        res = []

        def dfs(char):
            if char in visited:
                return visited[char]

            visited[char] = True

            for neighChar in adj[char]:
                if dfs(neighChar):
                    return True

            visited[char] = False
            res.append(char)

        for char in adj:
            if dfs(char):
                return ""

        res.reverse()
        return "".join(res)

    # 287. Find the Duplicate Number
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
            slow2 = 0
            while True:
                slow = nums[slow]
                slow2 = nums[slow2]
                if slow == slow2:
                    return slow

    # 295. Find Median from Data Stream
    class MedianFinder:

        def __init__(self):
            self.small, self.large = [], []

        def addNum(self, num: int) -> None:
            if self.large and num > self.large[0]:
                heapq.heappush(self.large, num)
            else:
                heapq.heappush(self.small, -1 * num)

            if len(self.small) > len(self.large) + 1:
                val = -1 * heapq.heappop(self.small)
                heapq.heappush(self.large, val)
            elif len(self.large) > len(self.small) + 1:
                val = heapq.heappop(self.large)
                heapq.heappush(self.small, -1 * val)

        def findMedian(self) -> float:
            if len(self.small) > len(self.large):
                return -1 * self.small[0]
            elif len(self.large) > len(self.small):
                return self.large[0]
            return (-1 * self.small[0] + self.large[0]) / 2

        # Your MedianFinder object will be instantiated and called as such:
        # obj = MedianFinder()
        # obj.addNum(num)
        # param_2 = obj.findMedian()

    # 297. Serialize and Deserialize Binary Tree
    class Codec:

        def serialize(self, root):
            res = []

            def dfs(node):
                if not node:
                    res.append("N")
                    return
                res.append(str(node.val))
                dfs(node.left)
                dfs(node.right)

            dfs(root)

            return ",".join(res)

        def deserialize(self, data):
            vals = data.split(",")
            self.i = 0

            def dfs():
                if vals[self.i] == "N":
                    self.i += 1
                    return None
                node = TreeNode(int(vals[self.i]))
                self.i += 1
                node.left = dfs()
                node.right = dfs()
                return node

            return dfs()

    # 300. Longest Increasing Subsequence
    def lengthOfLIS(self, nums: List[int]) -> int:
        LIS = [1] * len(nums)

        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 1, len(nums)):
                if nums[i] < nums[j]:
                    LIS[i] = max(LIS[i], 1 + LIS[j])

        return max(LIS)

    # 309. Best Time to Buy and Sell Stock with Cooldown
    def maxProfit(self, prices: List[int]) -> int:
        hm = {}

        def dfs(i, buying):
            if i >= len(prices):
                return 0
            if (i, buying) in hm:
                return hm[(i, buying)]
            cooldown = dfs(i + 1, buying)
            if buying:
                buy = dfs(i + 1, not buying) - prices[i]
                hm[(i, buying)] = max(buy, cooldown)
            else:
                sell = dfs(i + 2, not buying) + prices[i]
                hm[(i, buying)] = max(sell, cooldown)
            return hm[(i, buying)]
        return dfs(0, True)

    # 312. Burst Balloons
    def maxCoins(self, nums: List[int]) -> int:
        nums = [1] + nums + [1]
        cache = {}

        def dfs(l, r):
            if l > r:
                return 0
            if (l, r) in cache:
                return cache[(l, r)]
            cache[(l, r)] = 0
            for i in range(l, r + 1):
                coins = nums[l - 1] * nums[i] * nums[r + 1]
                coins += dfs(l, i - 1) + dfs(i + 1, r)
                cache[(l, r)] = max(cache[(l, r)], coins)
            return cache[(l, r)]
        return dfs(1, len(nums) - 2)

    # 322. Coin Change
    def coinChange(self, coins: List[int], amount: int) -> int:
        m = [amount + 1] * (amount + 1)
        m[0] = 0

        for a in range(1, amount + 1):
            for c in coins:
                if a - c >= 0:
                    m[a] = min(m[a], 1 + m[a - c])

        return m[amount] if m[amount] != amount + 1 else -1

    # 323. Number of component in an Undirected Graph
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        dsu = UnionFind()
        for a, b in edges:
            dsu.union(a, b)
        return len(set(dsu.findParent(x) for x in range(n)))

    # 329. Longest Increasing Path in a Matrix
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        ROWS, COLS = len(matrix), len(matrix[0])
        hm = {}

        def dfs(r, c, prevVal):
            if r < 0 or r == ROWS or c < 0 or c == COLS or matrix[r][c] <= prevVal:
                return 0
            if (r, c) in hm:
                return hm[(r, c)]

            res = 1
            res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))
            res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))
            res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))
            hm[(r, c)] = res
            return res

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, -1)

        return max(hm.values())

    # 332. Reconstruct Itinerary
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        adj = {src:[] for src, dst in tickets}
        tickets.sort()
        for src, dst in tickets:
            adj[src].append(dst)
        res = ["JFK"]
        def dfs(src):
            if len(res) == len(tickets) + 1:
                return True
            if src not in adj:
                return False
            copy = list(adj[src])
            for idx, val in enumerate(copy):
                adj[src].pop(idx)
                res.append(val)

                if dfs(val): return True
                adj[src].insert(idx, val)
                res.pop()
            return False
        dfs("JFK")
        return res

    # 338. Counting Bits
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        offset = 1

        for i in range(1, n + 1):
            if offset * 2 == i:
                offset = i
            dp[i] = 1 + dp[i - offset]

        return dp

    # 347. Top K Frequent Elements
    def topKFrequent(self, nums: list[int], k: int) -> list[
        int]:  # type: ignore
        count = {}
        freq = [[] for i in range(len(nums) + 1)]
        res = []
        for num in nums:
            count[num] = 1 + count.get(num, 0)
        for key, val in count.items():
            freq[val].append(key)
        for num in range(len(freq) - 1, 0, -1):
            for each in freq[num]:
                res.append(each)
                if len(res) == k:
                    return res

    # 355. Design Twitter
    class Twitter:

        def __init__(self):
            self.count = 0
            self.tweets_map = defaultdict(list)
            self.follower_map = defaultdict(set)

        def postTweet(self, userId: int, tweetId: int) -> None:
            self.tweets_map[userId].append([self.count, tweetId])
            self.count -= 1

        def getNewsFeed(self, userId: int) -> List[int]:
            res = []
            heap = []

            self.follower_map[userId].add(userId)

            for followee in self.follower_map[userId]:
                if followee in self.tweets_map:
                    idx = len(self.tweets_map[followee]) - 1
                    count, tweetID = self.tweets_map[followee][idx]
                    heap.append([count, tweetID, followee, idx - 1])

            heapq.heapify(heap)

            while heap and len(res) < 10:
                _, tweetID, followee, idx = heapq.heappop(heap)
                res.append(tweetID)
                if idx >= 0:
                    count, tweetID = self.tweets_map[followee][idx]
                    heapq.heappush(heap, [count, tweetID, followee, idx - 1])

            return res


        def follow(self, followerId: int, followeeId: int) -> None:
            self.follower_map[followerId].add(followeeId)

        def unfollow(self, followerId: int, followeeId: int) -> None:
            if followeeId in self.follower_map[followerId]:
                self.follower_map[followerId].remove(followeeId)


        # Your Twitter object will be instantiated and called as such:
        # obj = Twitter()
        # obj.postTweet(userId,tweetId)
        # param_2 = obj.getNewsFeed(userId)
        # obj.follow(followerId,followeeId)
        # obj.unfollow(followerId,followeeId)

    # 371. Sum of Two Integers
    def getSum(self, a: int, b: int) -> int:
        def add(a, b):
            if not a or not b:
                return a or b
            return add(a ^ b, (a & b) << 1)

        if a * b < 0:  # assume a < 0, b > 0
            if a > 0:
                return self.getSum(b, a)
            if add(~a, 1) == b:  # -a == b
                return 0
            if add(~a, 1) < b:  # -a < b
                return add(~add(add(~a, 1), add(~b, 1)), 1)  # -add(-a, -b)

        return add(a, b)  # a*b >= 0 or (-a) > b > 0

    # 416. Partition Equal Subset Sum
    def canPartition(self, nums: List[int]) -> bool:
        if sum(nums) % 2:
            return False
        dp = set()
        dp.add(0)
        target = sum(nums) / 2

        for i in range(len(nums) -1, -1, -1):
            worker = set()
            for t in dp:
                if t + nums[i] == target:
                    return True
                worker.add(t + nums[i])
                worker.add(t)
            dp = worker
        return True if target in dp else False

    # 417. Pacific Atlantic Water Flow
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        ROWS, COLS = len(heights), len(heights[0])
        pacific, atlantic = set(), set()

        def dfs(r, c, visited, previous_height):
            if ((r, c) in visited or r < 0 or c < 0 or r == ROWS or c == COLS
                    or heights[r][c] < previous_height):
                return
            visited.add((r, c))
            dfs(r + 1, c, visited, heights[r][c])
            dfs(r - 1, c, visited, heights[r][c])
            dfs(r, c + 1, visited, heights[r][c])
            dfs(r, c - 1, visited, heights[r][c])

        for c in range(COLS):
            dfs(0, c, pacific, heights[0][c])
            dfs(ROWS - 1, c, atlantic, heights[ROWS - 1][c])

        for r in range(ROWS):
            dfs(r, 0, pacific, heights[r][0])
            dfs(r, COLS - 1, atlantic, heights[r][COLS - 1])

        res = []
        for r in range(ROWS):
            for c in range(COLS):
                if (r, c) in pacific and (r, c) in atlantic:
                    res.append([r, c])

        return res

    # 424. Longest Repeating Character Replacement
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0
        l = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            while (r - l + 1) - max(count.values()) > k:
                count[s[l]] -= 1
                l = + 1
            res = max(res, r - l + 1)
        return res

    # optimized
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        res = 0
        l = 0
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])
            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        return res

    # 435. Non-overlapping Intervals
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        res = 0
        previous_end = intervals[0][-1]
        for start, end in intervals[1:]:
            if start >= previous_end:
                previous_end = end
            else:
                res += 1
                previous_end = min(end, previous_end)
        return res

    # 494. Target Sum
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        hm = {}

        def backtrack(idx, total):
            if idx == len(nums):
                return 1 if total == target else 0
            if (idx, total) in hm:
                return hm[(idx, total)]
            hm[(idx, total)] = backtrack(idx + 1, total + nums[idx]) + backtrack(idx + 1, total - nums[idx])

            return hm[(idx, total)]

        return backtrack(0, 0)

    # 518. Coin Change II
    def change(self, amount: int, coins: List[int]) -> int:
        # MEMOIZATION
        # Time: O(n*m)
        # Memory: O(n*m)
        cache = {}

        def dfs(i, a):
            if a == amount:
                return 1
            if a > amount:
                return 0
            if i == len(coins):
                return 0
            if (i, a) in cache:
                return cache[(i, a)]

            cache[(i, a)] = dfs(i, a + coins[i]) + dfs(i + 1, a)
            return cache[(i, a)]

        return dfs(0, 0)

        # DYNAMIC PROGRAMMING
        # Time: O(n*m)
        # Memory: O(n*m)
        dp = [[0] * (len(coins) + 1) for i in range(amount + 1)]
        dp[0] = [1] * (len(coins) + 1)
        for a in range(1, amount + 1):
            for i in range(len(coins) - 1, -1, -1):
                dp[a][i] = dp[a][i + 1]
                if a - coins[i] >= 0:
                    dp[a][i] += dp[a - coins[i]][i]
        return dp[amount][0]

        # DYNAMIC PROGRAMMING
        # Time: O(n*m)
        # Memory: O(n) where n = amount
        dp = [0] * (amount + 1)
        dp[0] = 1
        for i in range(len(coins) - 1, -1, -1):
            nextDP = [0] * (amount + 1)
            nextDP[0] = 1

            for a in range(1, amount + 1):
                nextDP[a] = dp[a]
                if a - coins[i] >= 0:
                    nextDP[a] += nextDP[a - coins[i]]
            dp = nextDP
        return dp[amount]

    # 543. Diameter of Binary Tree
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = [0]

        def dfs(root):
            if not root:
                return -1
            left = dfs(root.left)
            right = dfs(root.right)

            res[0] = max(res[0], left + right + 2)

            return 1 + max(left, right)

        dfs(root)

        return res[0]

    # 567. Permutation in String
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        s1count, s2count = [0] * 26, [0] * 26
        matches = 0
        l = 0
        for i in range(len(s1)):
            s1count[ord(s1[i]) - ord('a')] += 1
            s2count[ord(s2[i]) - ord('a')] += 1
        for i in range(26):
            matches += (1 if s1count[i] == s2count[i] else 0)
        for r in range(len(s1), len(s2)):
            if matches == 26:
                return True

            index = ord(s2[r]) - ord('a')
            s2count[index] += 1
            if s1count[index] == s2count[index]:
                matches += 1
            elif s1count[index] + 1 == s2count[index]:
                matches -= 1

            index = ord(s2[l]) - ord('a')
            s2count[index] -= 1
            if s1count[index] == s2count[index]:
                matches += 1
            elif s1count[index] - 1 == s2count[index]:
                matches -= 1
            l += 1
        return matches == 26

    # 572. Subtree of Another Tree
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not subRoot: return True
        if not root: return False

        if self.isSametree(root, subRoot):
            return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # 621. Task Scheduler
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = Counter(tasks)
        heap = [-i for i in count.values()]
        heapq.heapify(heap)
        time = 0
        q = deque()

        while heap or q:
            time += 1

            if not heap:
                time = q[0][1]
            else:
                cnt = 1 + heapq.heappop(heap)
                if cnt:
                    q.append([cnt, time + n])
            if q and q[0][1] == time:
                heapq.heappush(heap, q.popleft()[0])
        return time

    # 647. Palindromic Substrings
    def countSubstrings(self, s: str) -> int:
        res = 0
        for i in range(len(s)):
            l = r = i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                res += 1
                r += 1
                l -= 1

            l, r = i, i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                res += 1
                r += 1
                l -= 1

        return res

    # condensed with helper function
    def countSubstrings(self, s: str) -> int:
        res = 0

        for i in range(len(s)):
            res += self.countPali(s, i, i)
            res += self.countPali(s, i, i + 1)
        return res

    def countPali(self, s, l, r):
        res = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            res += 1
            l -= 1
            r += 1
        return res

    # 659. Encode and Decode Strings
    def encode(self, strs: list[str]) -> str:
        return "".join(f"#{len(i)}{i}" for i in strs)

    def decode(self, s: str) -> list[str]:
        res = []
        for idx, val in enumerate(s):
            if val == "#":
                g = s[idx + 2:idx + 2 + int(s[idx + 1])]
                if isinstance(int(s[idx + 1]), int):
                    res.append(g)
        return res

    # 663. Walls and Gates
    def walls_and_gates(self, rooms: List[List[int]]):
        ROWS, COLS = len(rooms), len(rooms[0])
        visited = set()
        q = deque()

        def addRoom(r, c):
            if (r < 0 or r == ROWS or c < 0 or c == COLS or (r, c) in visited or rooms[r][c] == -1):
                return
            visited.add((r, c))
            q.append([r, c])

        for r in range(ROWS):
            for c in range(COLS):
                if rooms[r][c] == 0:
                    q.append([r, c])
                    visited.add((r, c))

        distance = 0
        while q:
            for i in range(len(q)):
                r, c = q.popleft()
                rooms[r][c] = distance
                addRoom(r + 1, c)
                addRoom(r - 1, c)
                addRoom(r, c + 1)
                addRoom(r, c - 1)

            distance += 1

    # 678. Valid Parenthesis String
    def checkValidString(self, s: str) -> bool:
        lmin = lmax = 0

        for c in s:
            if c == "(":
                lmin += 1
                lmax += 1
            elif c == ")":
                lmin -= 1
                lmax -= 1
            else:
                lmin -= 1
                lmax += 1
            if lmax < 0:
                return False
            if lmin < 0:
                lmin = 0

        return lmin == 0

    # 684. Redundant Connection
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        parent = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)

        def find(n):
            p = parent[n]

            while p != parent[p]:
                parent[p] = parent[parent[p]]
                p = parent[p]

            return p

        def union(n1, n2):
            p1, p2 = find(n1), find(n2)

            if p1 == p2:
                return False

            if rank[p1] > rank[p2]:
                parent[p2] = p1
                rank[p1] += rank[p2]
            else:
                parent[p1] = p2
                rank[p2] += rank[p1]
            return True

        for n1, n2 in edges:
            if not union(n1, n2):
                return [n1, n2]

    # 695. Max Area of Island
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visited = set()
        result = 0

        def dfs(r, c):
            if (r < 0 or r == ROWS or c < 0 or c == COLS or
                    grid[r][c] == 0 or (r, c) in visited):
                return 0
            visited.add((r, c))
            return (1 + dfs(r + 1, c)
                    + dfs(r - 1, c)
                    + dfs(r, c + 1)
                    + dfs(r, c - 1))

        for r in ROWS:
            for c in COLS:
                result = max(dfs(r, c), result)

        return result

    # 703. Kth Largest Element in a Stream
    class KthLargest:

        def __init__(self, k: int, nums: List[int]):
            self.minheap = nums
            self.k = k
            heapq.heapify(self.minheap)
            while len(self.minheap) > k:
                heapq.heappop(self.minheap)

        def add(self, val: int) -> int:
            heapq.heappush(self.minheap, val)
            if len(self.minheap) > self.k:
                heapq.heappop(self.minheap)
            return self.minheap[0]

    # 704. Binary Search
    def search(self, nums: list[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            m = (r + l) // 2
            if nums[m] > target:
                r = m - 1
            elif nums[m] < target:
                l = m + 1
            else:
                return m
        return -1

    def search2(self, nums: list[int], target: int) -> int:
        mid = len(nums) // 2
        res = mid
        while nums[mid] != target:
            if mid > target:
                nums = nums[:mid]
                mid = len(nums) // 2
                res += mid
            elif mid == 0:
                return -1
            else:
                nums = nums[mid:]
                mid = len(nums) // 2
                res += mid
        return res

    # 739. Daily Temperatures
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        res = [0] * len(temperatures)
        stack = []

        for idx, temp in enumerate(temperatures):
            while stack and temp > stack[-1][0]:
                stackT, stackI = stack.pop()
                res[stackI] = (idx - stackI)
            stack.append([temp, idx])
        return res

    # 743. Network Delay Time
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edges = collections.defaultdict(list)
        for u, v, w in times:
            edges[u].append((v, w))
        minHeap = [(0, k)]
        visited = set()
        res = 0

        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in visited:
                continue
            visited.add(n1)
            res = max(res, w1)

            for n2, w2 in edges[n1]:
                if n2 not in visited:
                    heapq.heappush(minHeap, (w1 + w2, n2))

        return res if len(visited) == n else -1

    # 746. Min Cost Climbing Stairs
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)
        for i in range(len(cost) - 3, -1, -1):
            cost[i] += min(cost[i + 1], cost[i + 2])

        return min(cost[0], cost[1])

    # 763. Partition Labels
    def partitionLabels(self, s: str) -> List[int]:
        idxmap = {}

        for i, c in enumerate(s):
            idxmap[c] = i

        res = []
        size = 0
        end = -1

        for i, c in enumerate(s):
            size += 1
            end = max(end, idxmap[c])

            if i == end:
                res.append(size)
                size = 0

        return res

    # 778. Swim in Rising Water
    def swimInWater(self, grid: List[List[int]]) -> int:
        N = len(grid)
        visited = {(0, 0)}
        minHeap = [[grid[0][0], 0, 0]]
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while minHeap:
            t, r, c = heapq.heappop(minHeap)
            if r == N - 1 and c == N - 1:
                return t
            for dr, dc in directions:
                neiR, neiC = r + dr, c + dc
                if neiR < 0 or neiC < 0 or neiC == N or neiR == N or (neiR, neiC) in visited:
                    continue
                visited.add((neiR, neiC))
                heapq.heappush(minHeap, [max(t, grid[neiR][neiC]), neiR, neiC])

    # 787. Cheapest Flights Within K Stops
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0
        for i in range(k + 1):
            tmpPrice = prices.copy()
            for s, d, p in flights:  # source, destination, price
                if prices[s] == float("inf"):
                    continue
                if prices[s] + p < tmpPrice[d]:
                    tmpPrice[d] = prices[s] + p
            prices = tmpPrice
        return -1 if prices[dst] == float("inf") else prices[dst]

    # 846. Hand of Straights
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize:
            return False

        count = {}
        for i in hand:
            count[i] = 1 + count.get(i, 0)

        minH = list(count.keys())
        heapq.heapify(minH)

        while minH:
            n = minH[0]

            for i in range(n, n + groupSize):
                if i not in count:
                    return False
                count[i] =- 1
                if count[i] == 0:
                    if i != minH[0]:
                        return False
                    heapq.heappop(minH)

        return True

    # 853. Car Fleet
    def carFleet(self, target: int, position: list[int],
                 speed: list[int]) -> int:
        stack = []
        chain = sorted(zip(position, speed), reverse=True)
        for car in chain:
            tta = (target - car[0]) / car[1]
            if stack and stack[-1] < tta:
                stack.append(tta)
            elif not stack:
                stack.append(tta)
        return len(stack)

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: list[int], h: int) -> int:
        smin, smax = 1, max(piles)
        res = smax
        if smin >= smax:
            return smin
        else:
            while smin <= smax:
                candidate = (smin + smax) // 2
                time_req = sum([math.ceil(i / candidate) for i in piles])
                if time_req <= h:
                    res = min(res, candidate)
                    smax = candidate - 1
                elif time_req > h:
                    smin = candidate + 1
                print(f"{time_req = } | {candidate = } | ")
                print(f"{smin = } | {smax = } | ")
        return res

    # 919. Meeting Rooms II
    def min_meeting_rooms(self, intervals: List[Interval]) -> int:
        start = [i.start for i in intervals].sort()
        end = [i.end for i in intervals].sort()

        res, count = 0, 0
        s, e = 0, 0

        while s < len(intervals):
            if start[s] < end[s]:
                s += 1
                count += 1
            else:
                e += 1
                count -= 1
            res = max(res, count)

        return res

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        time = []
        for start, end in intervals:
            time.append((start, 1))
            time.append((end, -1))

        time.sort(key=lambda x: (x[0], x[1]))

        count = 0
        max_count = 0
        for t in time:
            count += t[1]
            max_count = max(max_count, count)
        return max_count

    # 920. Meeting Rooms
    def can_attend_meetings(self, intervals: List[Interval]) -> bool:
        intervals.sort(key=lambda x: x.start)

        for i in range(1, len(intervals)):
            l1 = intervals[i - 1]
            l2 = intervals[i]

            if l1.end > l2.start:
                return False

        return True

    # 953. Verifying an Alien Dictionary
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        orderID = {c: i for i, c in enumerate(order)}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            for j in range(len(w1)):
                if j == len(w2):
                    return False
                if w1[j] != w2[j]:
                    if orderID[w2[j]] < orderID[w1[j]]:
                        return False
                    break

        return True

    # 973. K Closest Points to Origin
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        minheap = []
        for x, y in points:
            dist = (x ** 2) + (y ** 2)
            minheap.append([dist, x, y])

        heapq.heapify(minheap)
        res = []
        while k > 0:
            _, x, y = heapq.heappop(minheap)
            res.append([x, y])
            k -= 1
        return res

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key=lambda k: k[0]*k[0]+k[1]*k[1])
        return points[:k]

    # 981. Time Based Key-Value Store
    class TimeMap:

        def __init__(self):
            self.store = {}  # ket=string, val=[list of [list of str, timestamp]]

        def set(self, key: str, value: str, timestamp: int) -> None:
            if key not in self.store:
                self.store[key] = []
            self.store[key].append([value, timestamp])

        def get(self, key: str, timestamp: int) -> str:
            res = ""
            values = self.store.get(key, [])
            left, right = 0, len(values) - 1
            while left <= right:
                mid = (left + right) // 2

                if values[mid][1] <= timestamp:
                    res = values[mid][0]
                    left = mid + 1
                else:
                    right = mid - 1
            return res

    # 994. Rotting Oranges
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        time, fresh = 0, 0
        ROWS, COLS = len(grid), len(grid[0])

        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1:
                    fresh += 1
                elif grid[r][c] == 2:
                    q.append([r, c])

        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        while q and fresh > 0:
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    row, col = r + dr, c + dc
                    if (row < 0 or col < 0 or row == ROWS or col == COLS or grid[row][col] != 1):
                        continue
                    grid[row][col] = 2
                    q.append([row, col])
                    fresh -= 1
            time += 1

        return time if not fresh else -1

    # 1046. Last Stone Weight
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = [-stone for stone in stones]
        heapq.heapify(heap)
        while heap and len(heap) > 1:
            stone1 = heapq.heappop(heap) # -y
            stone2 = heapq.heappop(heap) # -x
            res_stone = stone1 - stone2 # - (y - x)
            if res_stone:
                heapq.heappush(heap, res_stone)
        return -heap[0] if heap else 0

    # 1143. Longest Common Subsequence
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        grid = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]
        for i in range(len(text1) - 1, -1, -1):
            for j in range(len(text2) -1, -1, -1):
                if text1[i] == text2[j]:
                    grid[i][j] = 1 + grid[i + 1][j + 1]
                else:
                    grid[i][j] = max(grid[i + 1][j], grid[i][j + 1])
        return grid[0][0]

    # 1448. Count Good Nodes in Binary Tree
    def goodNodes(self, root: TreeNode) -> int:

        def dfs(node, maxV):
            if not node:
                return 0

            res = 1 if node.val >= maxV else 0
            maxV = max(maxV, node.val)

            res += dfs(node.left, maxV)
            res += dfs(node.right, maxV)

            return res

        return 1 + dfs(root.left, root.val) + dfs(root.right, root.val)

    # 1584. Min Cost to Connect All Points
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        N = len(points)

        adj = {i:[] for i in range(N)}
        for i in range(N):
            x1, y1 = points[i]
            for j in range(i + 1, N):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                adj[i].append([dist, j])
                adj[j].append([dist, i])

        res = 0
        visited = set()
        minHeap = [[0, 0]]
        while len(visited) < N:
            cost, i = heapq.heappop(minHeap)
            if i in visited:
                continue
            res += cost
            visited.add(i)
            for neiCost, nei in adj[i]:
                if nei not in visited:
                    heapq.heappush(minHeap, [neiCost, nei])

        return res

    # 1851. Minimum Interval to Include Each Query
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        intervals.sort()
        heap = []
        res = {}
        i = 0
        for q in sorted(queries):
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(heap, (r - l + 1, r))
                i += 1
            while heap and heap[0][1] < q:
                heapq.heappop(heap)
            res[q] = heap[0][0] if heap else -1

        return [res[q] for q in queries]

    # 1899. Merge Triplets to Form Target Triplet
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        available = set()
        for t in triplets:
            if (t[0] > target[0]
                    or t[1] > target[1]
                    or t[2] > target[2]):
                continue
            for i, v in enumerate(t):
                if v == target[i]:
                    available.add(i)

        return len(available) == 3

    # 2013. Detect Squares
    class DetectSquares:

        def __init__(self):
            self.ptsCount = defaultdict(int)
            self.pts = []

        def add(self, point: List[int]) -> None:
            self.ptsCount[tuple(point)] += 1
            self.pts.append(point)

        def count(self, point: List[int]) -> int:
            res = 0
            px, py = point
            for x, y in self.pts:
                if abs(py - y) != abs(px - x) or x == px or y == py:
                    continue
                res += self.ptsCount[(x, py)] * self.ptsCount[(px, y)]
            return res

    # Your DetectSquares object will be instantiated and called as such:
    # obj = DetectSquares()
    # obj.add(point)
    # param_2 = obj.count(point)