
#include "wsp.h"

/**********************/
/*** GeneratePoints ***/
/**********************/

point *GeneratePoints(int num, int dim){
  int i, j;
  double x, y, r, r_new, r_scale;
  bool found;
  point *point_set;
  double *temp_coord, temp_sum;

  point_set = (point *) malloc(sizeof(point) * num);
  for (i = 0; i < num; i++) {
    point_set[i].coord = (double *) malloc(sizeof(double)*dim);
  }

  /* Kuzmin model */
  temp_coord = (double *) malloc(sizeof(double) * dim);
  for (i=0; i<num; i++) {
    found = false;
    while (found == false) {
      temp_sum = 0.0;
      for (j=0; j<dim; j++) {
	temp_coord[j] = 2*(drand48() - .5);
	temp_sum += temp_coord[j]*temp_coord[j];
      }
      if (temp_sum <= 1)
	found = true;
    }
    r = sqrt(temp_sum);
    r_scale = 1.0;
    for (j=0; j<dim; j++) {
      r_scale *= (1-r);
    }
    for (j=0; j<dim; j++) {
      point_set[i].coord[j] = (temp_coord[j]/r)*((1.0/r_scale)-1.0);
    }
    point_set[i].lists = (list_elt **) malloc(sizeof(list_elt *) * dim);
  }
  
  return point_set;
}
