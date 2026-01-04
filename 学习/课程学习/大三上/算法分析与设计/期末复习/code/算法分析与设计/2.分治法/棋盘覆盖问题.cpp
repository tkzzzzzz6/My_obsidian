
/*
棋盘覆盖问题（L 型骨牌覆盖）伪代码：

tr: 棋盘左上角方格的行号
tc: 棋盘左上角方格的列号
dr: 特殊方格（缺口）的行号
dc: 特殊方格（缺口）的列号
size: 2^k，棋盘规模为 size * size

board[r][c] 记录该格被哪一块 L 型骨牌覆盖；tile 为全局递增编号。
*/

chessBoard(tr, tc, dr, dc, size) if size == 1 return

    t = tile++ s = size / 2

                   // 1) 左上子棋盘 [tr, tr+s-1] x [tc, tc+s-1]
                   if (dr < tr + s && dc < tc + s)
                       chessBoard(tr, tc, dr, dc, s) else board[tr + s - 1][tc + s - 1] = t chessBoard(tr, tc, tr + s - 1, tc + s - 1, s)

    // 2) 右上子棋盘 [tr, tr+s-1] x [tc+s, tc+size-1]
    if (dr < tr + s && dc >= tc + s)
        chessBoard(tr, tc + s, dr, dc, s) else board[tr + s - 1][tc + s] = t chessBoard(tr, tc + s, tr + s - 1, tc + s, s)

    // 3) 左下子棋盘 [tr+s, tr+size-1] x [tc, tc+s-1]
    if (dr >= tr + s && dc < tc + s)
        chessBoard(tr + s, tc, dr, dc, s) else board[tr + s][tc + s - 1] = t chessBoard(tr + s, tc, tr + s, tc + s - 1, s)

    // 4) 右下子棋盘 [tr+s, tr+size-1] x [tc+s, tc+size-1]
    if (dr >= tr + s && dc >= tc + s)
        chessBoard(tr + s, tc + s, dr, dc, s) else board[tr + s][tc + s] = t chessBoard(tr + s, tc + s, tr + s, tc + s, s)