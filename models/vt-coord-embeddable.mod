/*********************************************
 * OPL 12.10.0.0 Model
 * Author: chris
 * Creation Date: Jul 26, 2021 at 1:45:40 AM
 *********************************************/



float M = ...;                          // Big M
int num_workers = ...;                  // Number of workers
int num_roles = ...;                    // Number of individual roles
int num_projects = ...;                 // Number of projects
range H = 1..num_workers;               // Set of workers
range I = 1..num_roles;                 // Set of roles
range J = 1..num_projects;              // Set of roles
float T = ...;                          // Duration (weeks) in planning horizon
float T_role[I] = ...;                  // Duration (weeks) of role i within planning horizon
int b[I, J] = ...;                      // = 1 if role i belongs to project j, 0 otherwise
float c_plus[H, I] = ...;               // Cost for assigning worker h to role i
float c_minus[H, I] = ...;              // Cost for un-assigning worker h away from role i
float c_penalty[H] = ...;               // Penalty cost of per underutilized hour for worker h
float d[I] = ...;                       // Number of weekly hours required on role i
int q[H, I] = ...;                      // = 1 if worker h qualified for role i; 0 otherwise
float r_plus[J] = ...;                  // Revenue for selecting project j
float r_minus[J] = ...;                 // Penalty for rejecting project j    
float t_plus[H] = ...;                  // Max weekly hours for worker h
float t_minus[H] = ...;                 // Min desired weekly hours for worker h
int u[I, I] = ...;                      // = 1 if role k intersects role i's start time temporally, 0 otherwise
int v[I, I]= ...;                       // = 1 if workers cannot be assigned to both roles i and k, 0 otherwise
int w[H, I] = ...;                      // = 1 if worker h assigned to role i in previous allocation, 0 otherwise

// Decision Variables
dvar float+ e[H];                       // Number of hours below desired min over planning horizon for worker h
dvar int+ s[J] in 0..1;                 // = 1 if project i selected, 0 otherwise
dvar int+ x[H, I] in 0..1;              // = 1 if worker h assigned to role i in current plan; 0 otherwise
dvar int+ y[H, I] in 0..1;              // = 1 if worker h moved off role i between previous and current plan; 0 otherwise

// Objective function components
dvar float total_revenue;               // Total project revenue
dvar float assignment_cost;             // Cost of worker assignments
dvar float reassignment_cost;           // Cost or worker reassignments
dvar float underutilization_cost;       // Cost of underutilizations
dvar float total_cost;                  // Total cost
dvar float total_profit;                // Total profit

// Objective
maximize total_profit;                  // Eqn. (1) from paper
 
// Constraints
constraints {
  sum(j in J) (((r_minus[j] + r_plus[j]) * s[j]) - r_minus[j]) == total_revenue; // Total revenue
  sum(h in H, i in I) (c_plus[h,i] * x[h,i]) == assignment_cost;                 // Assignment cost 
  sum(h in H, i in I) (c_minus[h,i] * y[h,i]) == reassignment_cost;              // Reassignment cost
  sum(h in H) (c_penalty[h] * e[h]) == underutilization_cost;                    // Underutilization cost      
  total_cost == assignment_cost + reassignment_cost + underutilization_cost;     // Total cost
  total_profit == total_revenue - total_cost;                                    // Total profit
  
  // Below: Paper constraints
  forall (j in J) {
    sum(h in H, i in I) (b[i,j] * x[h,i]) == sum(i in I) (b[i,j] * s[j]);         // Constraint 2
  }
  forall (h in H) {
    forall (i in I) {
      (sum(k in I)(u[i,k] * d[k] * x[h,k])) - (M * (1 - x[h,i])) <= t_plus[h];    // Constraint 3
      (sum(k in I)(v[i,k] * x[h,k])) - (M * (1 - x[h,i])) <= 0;                   // Constraint 4
       x[h,i] <= q[h,i];                                                          // Constraint 7
       x[h,i] + y[h,i] >= w[h,i];                                                 // Constraint 8
    }
    (sum(i in I) (T_role[i] * d[i] * x[h,i])) + e[h] >= (T * t_minus[h]);         // Constraint 5
  }
  forall (i in I) {
    sum(h in H) (x[h,i]) <= 1;                                                    // Constraint 6
  }
}


// Write the x solution values, so they an be used as y-values
// in future problem modification.
execute DISPLAY {
  writeln();
  write("total_profit = ");
  write(total_profit);
  writeln();  
  writeln(); 
  write("e = [");
  for(var h = 1; h <= num_workers; h ++) {
    write(e[h]);
    if(h < num_workers) { write(","); }
  }
  write("]");
  writeln();  
  writeln(); 
  write("x = [");
  for(var h = 1; h <= num_workers; h ++) {
    write("[");
    for(var i = 1; i <= num_roles; i ++) {
      write(x[h][i]);
      if(i < num_roles) { write(","); }        
    }     
    write("]");   
    if(h < num_workers) { write(","); }      
  }   
  write("]");
  writeln();  
  writeln(); 
  write("y = [");
  for(var h = 1; h <= num_workers; h ++) {
    write("[");
    for(var i = 1; i <= num_roles; i ++) {
      write(y[h][i]);
      if(i < num_roles) { write(","); }        
    }     
    write("]");   
    if(h < num_workers) { write(","); }      
  }   
  write("]");
  writeln();  
  writeln(); 
  write("s = [");
  for(var j = 1; j <= num_projects; j ++) {
    write(s[j]);
    if(j < num_projects) { write(","); }
  }
  write("]");
  writeln();
}  