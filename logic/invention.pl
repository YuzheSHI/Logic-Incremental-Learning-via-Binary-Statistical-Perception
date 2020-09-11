diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-succ(A,C),diff_3(C,B).
diff_3(A,B):-succ(A,C),diff_2(C,B).
diff_2(A,B):-succ(A,C),succ(C,B).
diff_1(A,B):-succ(A,B).


diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-succ(A,C),diff_3(C,B).
diff_3(A,B):-succ(A,C),diff_2(C,B).
diff_2(A,B):-succ(A,C),succ(C,B).


diff_2(A,B):-diff_3_1(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-succ(A,C),diff_3(C,B).
diff_3(A,B):-succ(A,C),diff_3_1(C,B).
diff_3_1(A,B):-succ(A,C),succ(C,B).


diff_3(A,B):-succ(A,C),diff_4_1(C,B).
diff_2(A,B):-diff_4_1(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-diff_4_1(A,C),diff_4_1(C,B).
diff_4_1(A,B):-succ(A,C),succ(C,B).


diff_4(A,B):-diff_5_1(A,B).
diff_3(A,B):-succ(A,C),diff_5_2(C,B).
diff_2(A,B):-diff_5_2(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_5_1(C,B).
diff_5_1(A,B):-diff_5_2(A,C),diff_5_2(C,B).
diff_5_2(A,B):-succ(A,C),succ(C,B).


diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-succ(A,C),diff_6_1(C,B).
diff_3(A,B):-diff_6_1(A,B).
diff_2(A,B):-diff_6_2(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-diff_6_1(A,C),diff_6_1(C,B).
diff_6_1(A,B):-succ(A,C),diff_6_2(C,B).
diff_6_2(A,B):-succ(A,C),succ(C,B).


diff_6(A,B):-diff_7_1(A,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_4(A,B):-succ(A,C),diff_7_2(C,B).
diff_3(A,B):-diff_7_2(A,B).
diff_2(A,B):-diff_7_3(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-succ(A,C),diff_7(C,B).
diff_7(A,B):-succ(A,C),diff_7_1(C,B).
diff_7_1(A,B):-diff_7_2(A,C),diff_7_2(C,B).
diff_7_2(A,B):-succ(A,C),diff_7_3(C,B).
diff_7_3(A,B):-succ(A,C),succ(C,B).


diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_8_1(C,B).
diff_4(A,B):-diff_8_1(A,B).
diff_3(A,B):-succ(A,C),diff_8_2(C,B).
diff_2(A,B):-diff_8_2(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_8(C,B).
diff_8(A,B):-diff_8_1(A,C),diff_8_1(C,B).
diff_8_1(A,B):-diff_8_2(A,C),diff_8_2(C,B).
diff_8_2(A,B):-succ(A,C),succ(C,B).


diff_8(A,B):-diff_9_1(A,B).
diff_7(A,B):-succ(A,C),diff_6(C,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_9_2(C,B).
diff_4(A,B):-diff_9_2(A,B).
diff_3(A,B):-succ(A,C),diff_9_3(C,B).
diff_2(A,B):-diff_9_3(A,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_9_2(A,C),diff_9_2(C,B).
diff_9_2(A,B):-diff_9_3(A,C),diff_9_3(C,B).
diff_9_3(A,B):-succ(A,C),succ(C,B).


diff_1(A,B):-succ(A,B).
diff_2(A,B):-diff_9_3(A,B).
diff_3(A,B):-succ(A,C),diff_9_3(C,B).
diff_4(A,B):-diff_9_2(A,B).
diff_5(A,B):-succ(A,C),diff_9_2(C,B).
diff_6(A,B):-diff_7_1(A,B).
diff_7(A,B):-succ(A,C),diff_7_1(C,B).
diff_7_1(A,B):-diff_9_3(A,C),diff_9_2(C,B).
diff_8(A,B):-diff_9_1(A,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_9_2(A,C),diff_9_2(C,B).
diff_9_2(A,B):-diff_9_3(A,C),diff_9_3(C,B).
diff_9_3(A,B):-succ(A,C),succ(C,B).


diff_8(A,B):-diff_9_1(A,B).
diff_6(A,B):-succ(A,C),diff_5(C,B).
diff_5(A,B):-succ(A,C),diff_9_2(C,B).
diff_7(A,B):-diff_9_2(A,C),diff_3(C,B).
diff_4(A,B):-diff_9_2(A,B).
diff_2(A,B):-diff_9_3(A,B).
diff_3(A,B):-succ(A,C),diff_9_3(C,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_9_2(A,C),diff_9_2(C,B).
diff_9_2(A,B):-diff_9_3(A,C),diff_9_3(C,B).
diff_9_3(A,B):-succ(A,C),succ(C,B).


diff_6(A,B):-diff_7_1(A,B).
diff_8(A,B):-diff_9_1(A,B).
diff_5(A,B):-succ(A,C),diff_4(C,B).
diff_3(A,B):-succ(A,C),diff_4_1(C,B).
diff_2(A,B):-diff_4_1(A,B).
diff_7(A,B):-succ(A,C),diff_7_1(C,B).
diff_7_1(A,B):-diff_4_1(A,C),diff_4(C,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_4(A,C),diff_4(C,B).
diff_4(A,B):-diff_4_1(A,C),diff_4_1(C,B).
diff_4_1(A,B):-succ(A,C),succ(C,B).


diff_1(A,B):-succ(A,B).
diff_4(A,B):-diff_9_2(A,B).
diff_5(A,B):-succ(A,C),diff_9_2(C,B).
diff_2(A,B):-diff_9_3(A,B).
diff_6(A,B):-diff_9_3(A,C),diff_9_2(C,B).
diff_7(A,B):-diff_9_2(A,C),diff_3(C,B).
diff_8(A,B):-diff_9_1(A,B).
diff_3(A,B):-succ(A,C),diff_9_3(C,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_9_2(A,C),diff_9_2(C,B).
diff_9_2(A,B):-diff_9_3(A,C),diff_9_3(C,B).
diff_9_3(A,B):-succ(A,C),succ(C,B).


diff_2(A,B):-diff_9_3(A,B).
diff_4(A,B):-diff_9_2(A,B).
diff_5(A,B):-succ(A,C),diff_9_2(C,B).
diff_6(A,B):-diff_9_3(A,C),diff_9_2(C,B).
diff_7(A,B):-diff_9_2(A,C),diff_3(C,B).
diff_8(A,B):-diff_9_1(A,B).
diff_3(A,B):-succ(A,C),diff_9_3(C,B).
diff_1(A,B):-succ(A,B).
diff_9(A,B):-succ(A,C),diff_9_1(C,B).
diff_9_1(A,B):-diff_9_2(A,C),diff_9_2(C,B).
diff_9_2(A,B):-diff_9_3(A,C),diff_9_3(C,B).
diff_9_3(A,B):-succ(A,C),succ(C,B).

