import json
import os
import datetime
import subprocess
import util as ut
#from minizinc import Instance, Model, Solver
#import miniznc as 


# Turns a value (either a 1D/2D array or non-array) into an indexed dict for Pyomo.
# Reference: https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/raw_dicts.html
# Examples: indexed_dict(3)  ==> {None: 3}
#           indexed_dict([])) ==> {None: []}
#           indexed_dict([1,2,3]) ==> {None: [1, 2, 3]}
#           indexed_dict([[1,2,3],[4,5,6]]) ==> {(0, 0): 1, (0, 1): 2, (0, 2): 3, (1, 0): 4, (1, 1): 5, (1, 2): 6}
def indexed_dict(val, start_index=0):
        if type(val) == type([]):
            if len(val) > 0 and type(val[0]) == type([]):
                return {(i + start_index, j + start_index):val[i][j] for i in range(len(val)) for j in range(len(val[i]))}
            else:
                return {i + start_index:val[i] for i in range(len(val))}
        return {None: val}


# This class provides a Python representation of a virtual team coordination problem.
class Problem:
    '''
    % MiniZinc Model inputs
    float: M;                          % Big M
    int: num_workers;                  % Number of workers
    int: num_roles;                    % Number of individual roles
    int: num_projects;                 % Number of projects
    set of int: H = 1..num_workers;    % Set of workers
    set of int: I = 1..num_roles;      % Set of roles
    set of int: J = 1..num_projects;   % Set of roles
    float: T;                      % Duration (weeks) in planning horizon
    array[I] of float: T_role;     % Duration (weeks) of role i within planning horizon
    array[I, J] of int: b;         % = 1 if role i belongs to project j, 0 otherwise
    array[H, I] of float: c_plus;  % Cost for assigning worker h to role i
    array[H, I] of float: c_minus; % Cost for un-assigning worker h away from role i
    array[H] of float: c_penalty;  % Penalty cost of per underutilized hour for worker h
    array[I] of float: d;          % Number of weekly hours required on role i
    array[H, I] of int: q;         % = 1 if worker h qualified for role i; 0 otherwise
    array[J] of float: r_plus;     % Revenue for selecting project j
    array[J] of float: r_minus;    % Penalty for rejecting project j    
    array[H] of float: t_plus;     % Max weekly hours for worker h
    array[H] of float: t_minus;    % Min desired weekly hours for worker h
    array[I, I] of int: u;         % = 1 if role k intersect role i's start time temporally, 0 otherwise
    array[I, I] of int: v;         % = 1 if workers cannot be assigned to both roles i and k, 0 otherwise
    array[H, I] of int: w;         % = 1 if worker h assigned to role i in previous allocation, 0 otherwise
    '''

    # H=[], I=[], J=[], T_role={}, b=[], c_plus={}, c_minus={}, c_penalty={}, d={},
    #                   q=[], r_plus={}, r_minus={}, t_plus={}, t_minus={}, u=[], v=[], w=[]
    def __init__(self, H=[], I=[], J=[], T=0, T_role={}, b=[], c_plus={}, c_minus={}, c_penalty={}, d={},
                       q=[], r_plus={}, r_minus={}, t_plus={}, t_minus={}, u=[], v=[], w=[], M=999999999):
        self.H = sorted(list(set(H)))   # A list of unique worker ID's
        self.I = sorted(list(set(I)))   # A list of unique role ID's
        self.J = sorted(list(set(J)))   # A list of unique project ID's
        self.T = T                      # The number of weeks/time units in the planning horizon
        self.T_role = T_role            # A dict of form {<role id>:<num weeks>} week/time unit durations for each role.
        self.b = b                      # A list of form [(<role id>, <project id>), ...] indicating role membership in a project
        self.c_plus = c_plus            # A dict of form {(<worker id>, <role id>):<cost>} indicating the cost of assigning a worker to a role
        self.c_minus = c_minus          # A dict of form {(<worker id>, <role id>):<cost>} indicating the cost of moving a worker off a role
        self.c_penalty = c_penalty      # A dict of form {<worker id>:<cost>} indicating the cost per week/time unit of underutilization
        self.d = d                      # A dict of form {<role id>:<time>} indicating the weekly/per time unit time demand for each role
        self.q = q                      # A list of form [(<worker id>, <role id>), ...] containing the qualification pairs
        self.r_plus = r_plus      # A dict of form {<project id>:<revenue>}        
        self.r_minus = r_minus    # A dict of form {<project id>:<rejection penalty cost>}
        self.t_plus = t_plus      # A dict of form {<worker id>:<time>} indicating the min. preferred weekly/per time unit hours
        self.t_minus = t_minus    # A dict of form {<worker id>:<time>} indicating the max possible weekly/per time unit hours
        self.u = u                # A list of form [(<role id 1>, <role id 2>), ...] indicating that role 2 overlaps role 1's start time
        self.v = v                # A list of form [(<role id 1>, <role id 2>), ...] indicating that no worker can be assigned to both
        self.w = w                # A list of form [(<worker id>, <role id>), ...] indicating that the worker was previously assigned the role
        self.M = M                # Big M

    # Gets a list of all problem field names.
    @property
    def field_names(self):
        fields = ['H', 'I', 'J', 'T', 'T_role', 'b', 'c_plus', 'c_minus', 'c_penalty', 'd',
                  'q', 'r_plus', 'r_minus', 't_plus', 't_minus', 'u', 'v', 'w', 'M']
        return fields

    # This builds a dict of OPL model inputs for this problem.
    def to_opl_dict(self):
        data = {'M':self.M, 'num_workers':len(self.H), 'num_roles':len(self.I), 'num_projects':len(self.J), 'T':self.T}
        data['T_role'] = [self.T_role[i] for i in self.I]
        sb = set(self.b)
        data['b'] = [[1 if (i, j) in sb else 0 for j in self.J] for i in self.I]
        data['c_plus'] = [[self.c_plus[(h, i)] if (h, i) in self.c_plus else 0 for i in self.I] for h in self.H]
        data['c_minus'] = [[self.c_minus[(h, i)] if (h, i) in self.c_minus else 0 for i in self.I] for h in self.H]
        data['c_penalty'] = [self.c_penalty[h] if h in self.c_penalty else 0 for h in self.H]
        data['d'] = [self.d[i] for i in self.I]
        sq = set(self.q)
        data['q'] = [[1 if (h, i) in sq else 0 for i in self.I] for h in self.H]
        data['r_plus'] = [self.r_plus[j] for j in self.J]
        data['r_minus'] = [self.r_minus[j] for j in self.J]
        data['t_plus'] = [self.t_plus[h] for h in self.H]
        data['t_minus'] = [self.t_minus[h] for h in self.H]
        su = set(self.u)
        data['u'] = [[1 if (i, k) in su else 0 for k in self.I] for i in self.I]
        sv = set(self.v)
        data['v'] = [[1 if (i, k) in sv else 0 for k in self.I] for i in self.I]
        sw = set(self.w)
        data['w'] = [[1 if (h, i) in sw else 0 for i in self.I] for h in self.H]
        return data

    # Convert to OPL data file string
    def to_opl_data(self, output_filename=None):
        s = ''
        for (param, val) in self.to_opl_dict().items():
            s += str(param) + ' = ' + str(val) + ";\n"
        if output_filename != None:
            ut.write_file(s, output_filename)
        return s
    
    # This builds a dict of Pyomo model inputs for this problem.
    # Example: https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/raw_dicts.html
    def build_pyomo_data(self):
        '''
        from pyomo.environ import *
        m = AbstractModel()
        m.I = Set()
        m.p = Param()
        m.q = Param(m.I)
        m.r = Param(m.I, m.I, default=0)
        data = {None: {
            'I': {None: [1,2,3]},
            'p': {None: 100},
            'q': {1: 10, 2:20, 3:30},
            'r': {(1,1): 110, (1,2): 120, (2,3): 230},
        }}
        i = m.create_instance(data)
        '''
        dd = self.to_opl_dict()
        return {None: {param_name:indexed_dict(param_val, start_index=1) for (param_name, param_val) in dd.items()} }
        #return {None:dd}
        
    # Get the projects that the given role_id belongs to.
    def get_role_projects(self, role_id):
        return list(set([y for (x, y) in self.b if x == role_id]))
    
    # Get the roles that belong to the given project_id.
    def get_project_roles(self, project_id):
        return list(set([x for (x, y) in self.b if y == project_id]))
    
    # Add new workers.
    def add_workers(self, worker_ids):
        self.H = sorted(list(set(self.H + worker_ids)))
    
    # Delete the specified workers.
    def delete_workers(self, worker_ids): 
        wids = set(worker_ids)
        self.c_plus = {(w, r):self.M if w in wids else self.c_plus[(w, r)] for (w, r) in self.c_plus}
        self.c_penalty = {w:(0 if w in wids else self.c_penalty[w]) for w in self.c_penalty}
        self.t_plus = {w:(0 if w in wids else self.t_plus[w]) for w in self.t_plus}
        self.t_minus = {w:(0 if w in wids else self.t_minus[w]) for w in self.t_minus}
        self.q = [(w, r) for (w, r) in self.q if not(w in wids)]
        
    # Delete the specified roles.
    def delete_roles(self, role_ids): 
        rids = set(role_ids)
        self.b = [(r, p) for (r, p) in self.b if not(r in rids)]
        self.c_plus = {(w, r):self.M if r in rids else self.c_plus[(w, r)] for (w, r) in self.c_plus}
        self.c_minus = {(w, r):(0 if r in rids else self.c_minus[(w, r)]) for (w, r) in self.c_minus}
        self.d = {r:(0 if r in rids else self.d[r]) for r in self.d}
        self.q = [(w, r) for (w, r) in self.q if not(r in rids)]
    
    # Delete the specified projects. Removes all roles that solely belong to one of these projects if
    # delete_member_roles=True.
    def delete_projects(self, project_ids, delete_member_roles=True): 
        pids = set(project_ids)
        self.r_minus = {p:(0 if p in pids else self.r_minus[p]) for p in self.r_minus}
        if delete_member_roles:
            member_roles = []
            for p in pids:
                member_roles += [(r, set(self.get_role_projects(r))) for r in self.get_project_roles(p)]
            to_go_roles = [r for (r, ps) in member_roles if len(ps) < 2]
            self.delete_roles(to_go_roles)
    
    # Add specific components to the problem.    
    def add_components(self, H=[], I=[], J=[], T_role={}, b=[], c_plus={}, c_minus={}, c_penalty={}, d={},
                       q=[], r_plus={}, r_minus={}, t_plus={}, t_minus={}, u=[], v=[], w=[]):
        self.H = sorted(list(set(H).union(self.H)))
        self.I = sorted(list(set(I).union(self.I)))
        self.J = sorted(list(set(J).union(self.J)))
        self.T_role = dict(list(self.T_role.items()) + list(T_role.items()))
        self.b += list(set(b))
        self.c_plus = dict(list(self.c_plus.items()) + list(c_plus.items()))
        self.c_minus = dict(list(self.c_minus.items()) + list(c_minus.items()))
        self.c_penalty = dict(list(self.c_penalty.items()) + list(c_penalty.items()))
        self.d = dict(list(self.d.items()) + list(d.items()))
        self.q += list(set(q))
        self.r_plus = dict(list(self.r_plus.items()) + list(r_plus.items()))
        self.r_minus = dict(list(self.r_minus.items()) + list(r_minus.items()))
        self.t_plus = dict(list(self.t_plus.items()) + list(t_plus.items()))
        self.t_minus = dict(list(self.t_minus.items()) + list(t_minus.items()))
        self.u += list(set(u))
        self.v += list(set(v))
        self.w += list(set(w))
    
    # Delete the specified components from the problem.    
    def delete_components(self, H=[], I=[], J=[], b=[], q=[], u=[], v=[], w=[], 
                          delete_project_member_roles=True): 
        self.delete_workers(H)
        self.delete_roles(I)
        self.delete_projects(J, delete_member_roles = delete_project_member_roles)
        self.b = [x for x in self.b if not(x in b)]
        self.q = [x for x in self.q if not(x in q)]
        self.u = [x for x in self.u if not(x in u)]
        self.v = [x for x in self.v if not(x in v)]
        self.w = [x for x in self.w if not(x in w)]
        
    # This updates (i.e. overwrites) the self.w model input (previous assignments) with a set of new assignments, (the new_w_input).
    # w = A list of form [(<worker id>, <role id>), ...] indicating that the worker was previously assigned the role
    # new_w_input is array[H, I] of int: w;  % = 1 if worker h assigned to role i in previous allocation, 0 otherwise
    # ** IMPORTANT NOTE: when updating, this MUST be called before any component additions/deletions to maintain consistency.
    def update_w_input(self, new_w_input):
        w = []
        for (h, row) in enumerate(new_w_input):
            for (i, val) in enumerate(row):
                if val == 1:
                    w.append((self.H[h], self.I[i]))
        self.w = w
    
    # Convert an minizinc x variable matrix to assignments as a list in form [(<worker id>, <role id>), ...]
    def get_assignments(self, x_var):
        a = []
        for (h, row) in enumerate(x_var):
            for (i, val) in enumerate(row):
                if val == 1:
                    a.append((self.H[h], self.I[i]))
        return a
    
    # Convert a minizinc s variable array to a list of selected projects.
    def get_selected_projects(self, s_var):
        s = []
        for (j, val) in enumerate(s_var):
            if val == 1:
                s.append(self.J[j])
        return s
    
    # Get a clone of this Problem object.    
    def clone(self):
        return Problem(self.H, self.I, self.J, self.T, self.T_role, self.b, self.c_plus, 
                       self.c_minus, self.c_penalty, self.d, self.q, self.r_plus, self.r_minus, 
                       self.t_plus, self.t_minus, self.u, self.v, self.w, self.M)
    
    # Get a string representation of this Problem object.                  
    def to_s(self):
        s = ''
        for (key, val) in self.to_dict().items():
            s += str(key) + ' = ' + str(val) + ';\n'
        return s
    
    # Mkae to_s compatible with the standatd Python string method.
    def __str__(self):
        return self.to_s()
    
    # Get a dict representation of this Problem object.    
    def to_dict(self):
        return {f:getattr(self, f) for f in self.field_names}
    
    # Construct a Problem from its dict representation.    
    def from_dict(self, dct):
        for (key, val) in dct.items():
            try:
                setattr(self, key, val)
            except:
                print('No field able to be assigned:', (key, val))
                
    # Save to an output file in JSON format
    def to_json_file(self, output_filename):
        with open(output_filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile)
            
    # Construct a Problem from an JSON file.    
    def from_json_file(self, input_filename):
        with open(input_filename, 'r') as infile:
            self.from_dict(json.load(infile))
    
    # Save to an output file in string format
    def to_text_file(self, output_filename):
        with open(output_filename, 'w') as outfile:
            outfile.write(str(self.to_dict()))
            
    # Construct a Problem from an JSON file.    
    def from_text_file(self, input_filename):
        with open(input_filename, 'r') as infile:
            self.from_dict(eval(infile.read()))
    
    
        

# This class uses OPL models together with oplrun.exe to get solutions.
class OplrunSolver:
    def __init__(self, model_file_path, temp_model_folder_path='./'):
        self.model_file = model_file_path
        self.temp_model_folder_path = temp_model_folder_path
    
    # Solve the specified problem with OPL, write the solution to the specified output file,
    # and return the results.
    def solve(self, problem, solution_filename=None, time_limit=None):   
        model_file = self.model_file
        delete_data_file = False
        if isinstance(problem, Problem):
            delete_data_file = True
            data_file = os.path.join(self.temp_model_folder_path, '___TMPDATA__.dat')
            problem.to_opl_data(data_file)
        else:
            data_file = problem
        if time_limit != None and self.temp_model_folder_path != None:
            mod_contents = ut.read_file(model_file)
            tl_stmt = "execute PARAMS {\n  cplex.tilim = " + str(time_limit) + "; // time limit in seconds \n}\n\n"
            mod_contents = tl_stmt + mod_contents
            model_file =  os.path.join(self.temp_model_folder_path, '___TLMOD__1.mod')
            ut.write_file(mod_contents, model_file)
        raw_output = subprocess.check_output(['oplrun', model_file, data_file]).decode('ascii')
        lines = raw_output.split("\n")
        if solution_filename != None:
            print('Writing solution file: ' + solution_filename)
            ut.write_file(raw_output, solution_filename)
        x, y, e, s, objective, time, gap = None, None, None, None, None, None, None
        
        try:
            text = [x for x in lines if x.startswith('x =')][0]
            x = eval(text.split('=')[1].strip())
        except:
            print('Error getting x: ' + str(solution_filename))
        try:
            text = [x for x in lines if x.startswith('y =')][0]
            y = eval(text.split('=')[1].strip())
        except:
            print('Error getting y: ' + str(solution_filename))
        try:
            text = [x for x in lines if x.startswith('e =')][0]
            e = eval(text.split('=')[1].strip())
        except:
            print('Error getting e: ' + str(solution_filename))
        try:
            text = [x for x in lines if x.startswith('s =')][0]
            s = eval(text.split('=')[1].strip())
        except:
            print('Error getting x: ' + str(solution_filename))
        try:
            obj_text = [x for x in lines if x.startswith('OBJECTIVE:')][0]
            objective = float(obj_text.split(':')[1].strip())
        except:
            print('Error getting objective: ' + str(solution_filename))
        try:
            if objective != None:
                time = 0
            time_text = [x for x in lines if x.startswith('Total')][0]
            time_list = time_text.split('=')
            time = eval(time_list[1].split('sec')[0].strip())
        except:
            print('Error getting time: ' + str(solution_filename))  
        try:
            for line in lines:
                try:
                    if '%' in line:
                        gt = line.split(' ')[-1].strip()
                        g = eval(gt[:len(gt) - 1])
                        gap = g if gap == None or g < gap else gap
                except:
                    pass
            if time_limit != None and time < time_limit:
                gap = 0
        except:
            print('Error getting MIP gap: ' + str(solution_filename))
        if time_limit != None and self.temp_model_folder_path != None:
            os.remove(model_file) 
        if delete_data_file:
            os.remove(data_file)
        h = {'x':x, 'y':y, 'e': e, 's':s, 'objective':objective, 'time':time, 'gap':gap if gap != None else 0}
        return h