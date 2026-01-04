#define MaxSize 100
typedef int ElemType;
typedef struct
{
    ElemType data[MaxSize];
    int length;
} Sqlist;

// 建立顺序表
void createList(Sqlist *&L, ElemType a[], int n)
{
    int i = 0, k = 0;
    L = (Sqlist *)malloc(sizeof(Sqlist));
    while (i < n)
    {
        L->data[k] = a[i];
        ++k;
        ++i;
    }
    L->length = k;
}
