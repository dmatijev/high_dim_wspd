
#include "wsp.h"


double upper_bound(int n, int d, double s)
{
  return 2.0*(n-1)*pow( (3.0*(s*sqrt(d)+2*sqrt(d)+1)+2) , d);
  //return 1.0;
}


/**********************/
/*** PrintPoints ***/
/**********************/
void PrintPoints(point *pts, int num, int dim) {
  int i, j;

  for(i = 0; i < num; i++) {
    printf("%d: (", i);
    for(j = 0; j < dim; j++) {
      printf("%lf, ", pts[i].coord[j]);
    }
    printf("\b\b)\n");
  }
}

/**************************/
/*** PrintOrderedPoints ***/
/**************************/
void PrintOrderedPoints(int dim, list_set orders) {
  int i, j, k;
  list_elt *elt, *elt1;

  for(i = 0; i < dim; i++) {
    printf("Dimension %d:\n", i);
    elt = orders[i].first;
    elt1 = orders[i].last;
    k = 0;
    do {
      printf("%d: (", k);
      for(j = 0; j < dim; j++) {
	printf("%0.2lf, ", elt->pt->coord[j]);      
      }
      printf("%x)\t", elt->pt->new_list);

      printf("%d: (", k);
      for(j = 0; j < dim; j++) {
	printf("%0.2lf, ", elt1->pt->coord[j]);      
      }
      printf("%x)\n", elt1->pt->new_list);

      elt = elt->next;
      elt1 = elt1->prev;
      k++;
    } while((elt != NULL) || (elt1 != NULL));
    printf("--------\n");
  }
}

/**************************/
/*** PrintTree ***/
/**************************/
void PrintTree(tree_node *root, int dim, int level) {
  int j, i;

  if(root->lchild == NULL) {
    for(i = 0; i < level * 2; i++) printf(" ");
    printf("(");
    for(j = 0; j < dim - 1; j++) 
      printf("%0.2lf, ", root->pt->coord[j]);
    printf("%0.2lf)LEAF\n", root->pt->coord[dim-1]);
  }
   
  else {
    for(i = 0; i < level * 2; i++) printf(" ");
    printf("<");
    for(j = 0; j < dim - 1; j++) 
      printf("(%0.2lf, ", root->center[j]);
    printf("%0.2lf), ", root->center[dim-1]);
    printf("%0.2lf>\n", root->radius);
    PrintTree(root->lchild, dim, level + 1);
    PrintTree(root->rchild, dim, level + 1);
  }
}

/**********************/
/*** ListEltCompare ***/
/**********************/

int ListEltCompare(const void *ple1, const void *ple2){
  int dim;
  list_elt *le1 = (list_elt *)(ple1), *le2 = (list_elt *)(ple2);
  dim = le1->pt->sort_dim;

  return ((le1->pt->coord[dim] - le2->pt->coord[dim]) > 0);
}

/******************/
/*** DoubleLink ***/
/******************/

void DoubleLink(list_elt *elt_array, int num, list *linked_list) {
  int i;
  
  for(i = 0; i < num; i++) {
    if(i == 0)
      elt_array[i].prev = NULL;
    else
      elt_array[i].prev = &(elt_array[i-1]);
    if(i == (num - 1))
      elt_array[i].next = NULL;
    else
      elt_array[i].next = &(elt_array[i+1]);
  }
  linked_list->first = &(elt_array[0]);
  linked_list->last = &(elt_array[num-1]);
}


/******************/
/*** SortPoints ***/
/******************/

void SortPoints(point *pts, int num, int dim, list *sorted_list){
  int i;
  list_elt *sorted_list_elt;

  sorted_list_elt = (list_elt *) malloc(sizeof(list_elt)*num);
  for (i=0; i<num; i++) {
    pts[i].sort_dim = dim;
    sorted_list_elt[i].pt = &(pts[i]);
  }

  qsort(sorted_list_elt, num, sizeof(list_elt), (ListEltCompare));

  DoubleLink(sorted_list_elt, num, sorted_list);

  for(i = 0; i < num; i++)
    sorted_list_elt[i].pt->lists[dim] = &(sorted_list_elt[i]);
}

/****************/
/*** ListCopy ***/
/****************/

/* note this does one important thing besides copying:
   it sets the new_list field to null for all points */

void ListCopy(list *l_orig, list *l_copy, int num){
  int i;
  list_elt *list_elt_copy, *current;

  list_elt_copy = (list_elt *) malloc(sizeof(list_elt) * num);

  current = l_orig->first;
  for (i=0; i<num; i++) {
    list_elt_copy[i].pt = current->pt;
    current->pt->new_list = NULL;
    current = current->next;
  }

  DoubleLink(list_elt_copy, num, l_copy);
}







