void LineSimple(int x0, int y0, int xn, int yn)
{
    int dy = yn - y0;
    int dx = xn - x0;
    double k = dy / dx;
    double b = y0 - k * x0;
    int x = x0;
    int y = y0;
    for (x = x0; x <= xn; x++)
    {
        y = k * x + b;
        putpixel(x, int(y + 0.5));
    }
}