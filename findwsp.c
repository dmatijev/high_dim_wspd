#include <fstream>
#include <assert.h>
#include <vector>
#include "wsp.h"

int FindWSP2(tree_node *tnode1, tree_node *tnode2, double s, int dim);
int wellsep(tree_node *tnode1, tree_node *tnode2, double s, 
            int dim, double& dist, double &actual_s);

extern std::ofstream output_file;

/***************/
/*** FindWSP ***/
/***************/

int FindWSP(tree_node *tnode, double s, int dim){

  if (tnode->pt != NULL) {
    /* printf("tnode->pt != NULL!!!\n"); */
    return (0);
  }
  else {
    return (FindWSP(tnode->lchild, s, dim)
	    + FindWSP(tnode->rchild, s, dim)
	    + FindWSP2(tnode->lchild, tnode->rchild, s, dim)); 
  }
}

int dfs(tree_node *tnode, vector<double*>& pt)
{
	if (tnode->lchild == NULL && tnode->rchild == NULL){
		pt.push_back(tnode->pt->coord);
		return 1;
	}
	return dfs(tnode->lchild, pt) + dfs(tnode->rchild, pt);
}

// EDIT: get index information
using Points = vector<point*>;

int dfs(tree_node *tnode, Points& pts)
{
	if (tnode->lchild == NULL && tnode->rchild == NULL){
		pts.push_back(tnode->pt);
		return 1;
	}
	return dfs(tnode->lchild, pts) + dfs(tnode->rchild, pts);
}


/****************/
/*** FindWSP2 ***/
/****************/

// EDIT:

void output_indices(Points &l, Points &r) {
  for (auto p : l) output_file << p->index << " ";
  output_file << " | ";
  for (auto p : r) output_file << p->index << " ";
  output_file << "\n";
}

void output_info(double sep, tree_node *l, tree_node *r, int dim) {
  for (int i=0; i<dim; i++) {
    output_file << l->center[i] << " "; 
  }
  output_file << " | ";
  for (int i=0; i<dim; i++) {
    output_file << r->center[i] << " "; 
  } 
  output_file << " | ";
}

int FindWSP2(tree_node *tnode1, tree_node *tnode2, double s, int dim) {
  double distance, actual_s;
  if (wellsep(tnode1, tnode2, s, dim, distance, actual_s)) {
    // EDIT:
	  Points tnode1_pts, tnode2_pts;
	  dfs(tnode1, tnode1_pts);
	  dfs(tnode2, tnode2_pts);
    output_info(actual_s,tnode1,tnode2, dim);
    output_indices(tnode1_pts,tnode2_pts);
	  return (1);
  }
  else {
    if (tnode1->radius > tnode2->radius)
      return (FindWSP2(tnode1->lchild, tnode2, s, dim) 
	      + FindWSP2(tnode1->rchild, tnode2, s, dim));
    else
      return (FindWSP2(tnode1, tnode2->lchild, s, dim) 
	      + FindWSP2(tnode1, tnode2->rchild, s, dim));
  }
}

/***************/
/*** wellsep ***/
/***************/

int wellsep(tree_node *tnode1, tree_node *tnode2, double s, 
            int dim, double& dist, double &actual_s){
  int i;
  double radius, distance;

  radius = std::max(tnode1->radius, tnode2->radius);

  distance = 0.0;
  for (i=0; i<dim; i++) {
    distance += (tnode1->center[i] - tnode2->center[i])*
      (tnode1->center[i] - tnode2->center[i]);
  }

  dist = distance = sqrt(distance);
  distance = distance - 2*radius;
  actual_s = radius == 0 ? -1 : distance / radius;

  if (distance > s*radius)
    return (1);
  else
    return (0);
}
