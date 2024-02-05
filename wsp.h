#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

#include "Timer.h"
using namespace std;

//#define max(a,b) ((a) >= (b) ? (a) : (b))

//typedef enum BOOL {true,false} bool;
typedef enum HALF {lower,upper} half;

/******************************************************************/

typedef struct List_elt{
  struct List_elt *next, *prev;      /* next/prev list element    */
  struct Point *pt;                  /* point represented         */
} list_elt;

typedef struct List{
  list_elt *mem;                     /* pointer to memory to free */
  list_elt *first, *last;            /* first and last elements   */
} list;                              /*    of sorted list         */

typedef list *list_set;              /* set of d=dim lists        */

typedef struct List_set_elt {
  list_set data;                     /* set of d=dim lists        */
  int count;                         /* how many items in this list */
  struct List_set_elt *next;         /* next list set             */
} list_set_elt;

typedef struct Point{
  double *coord;                     /* location of point         */
  list_set new_list;                 /* lists to be inserted into */
  list_elt **lists;                  /* where pt is currently at  */
  int sort_dim;                      /* dim along which points    */
  // EDIT: row of the point
  int index;
} point;                             /*    are being sorted       */

typedef struct Tree_node{
  struct Tree_node *lchild, *rchild; /* children of tree node     */
  double *center;                    /* center of pts contained   */
  double radius;                     /* radius containing all pts */
  point *pt;                         /* if leaf node, point       */
  int nr_pt;
} tree_node;                         /*    represented */

/******************************************************************/

point *GeneratePoints(int num, int dim);
void SortPoints(point *, int num, int dim, list *sorted_list);
void ListCopy(list *l_orig, list *l_copy, int num);
void DoubleLink(list_elt *elt_array, int num, list *linked_list);
tree_node *BuildTree(list_set ls, int num, int dim);

int FindWSP(tree_node *tnode, double s, int dim);
double upper_bound(int n, int d, double s);
