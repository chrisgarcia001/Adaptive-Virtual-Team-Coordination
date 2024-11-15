import random as rnd
import functools as ft
from core import *

'''
% Model inputs
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


# Generate a random float in the specified range.
def rand_float(minf, maxf):
    return minf + ((maxf - minf) * rnd.random())

# Given a list of numeric weights that correspond to the probability that an index will be chosen, randomly select
# one of the indices and return it.    
def random_weighted_index(weights):
    if len(weights) < 1:
        raise 'random_weight_index: weights must not be empty'
    sm = sum(weights)
    tw = 0
    rd = rand_float(0, sm)
    for i in range(len(weights)):
        tw += weights[i]
        if rd <= tw:
            return i

# For a list of items and a number k, get all unique item combinations of length k.           
def k_comb(items, k):
    if len(items) == 0:
        return []
    if k == len(items):
        return [items]
    if k == 1:
        return [[x] for x in items] 
    return [[items[0]] + h for h in k_comb(items[1:], k - 1)] + k_comb(items[1:], k)
 
# For an arbitrary input of sets or lists, get the cartesian product. 
def cartesian_prod(*sets):
    if len(sets) == 0:
        return []
    if len(sets) == 1:
        return [[x] for x in sets[0]]
    rest = cartesian_prod(*sets[1:])
    return [[y] + z for y in sets[0] for z in rest]

class Worker:
    # @param wid: a string designating the worker ID
    # @param qualifications: a list of worker qualifications (<type>, <max level>)
    # @param hourly_cost: the hourly cost of the worker
    # @param minmax_weekly_hours: a tuple of form (<min hours per week desired>, <max hours per week possible>)
    # @param underutilization_hourly_cost: the cost per hour underutilization of this worker
    def __init__(self, wid, qualifications, hourly_cost, minmax_weekly_hours, underutilization_hourly_cost):
        self.wid = wid
        self.qualifications = qualifications
        self.hourly_cost = hourly_cost
        self.minmax_weekly_hours = minmax_weekly_hours
        self.underutilization_hourly_cost = underutilization_hourly_cost
        
    def __str__(self):
        s = 'WID: ' + str(self.wid)
        s += "\nQualifications: " + str(self.qualifications)
        s += "\nHourly Cost: " + str(self.hourly_cost)
        s += "\nMin/Max Weekly Hours: " + str(self.minmax_weekly_hours) 
        s += "\nUnderutilization Hourly Cost: " + str(self.underutilization_hourly_cost) + "\n"
        return s


class Role:
    # @param rid: a string - the unique ID of this role.
    # @param skill: a tuple of form (family, level)
    # @param weekly_hours: an int designating the required hours per week for this role.
    # @param duration: an int specifying the number of weeks duration for this role.
    # @param start_time: an int specifying the starting week.
    def __init__(self, rid, skill, weekly_hours, duration=None, start_time=None):
        self.rid = rid
        self.skill = skill
        self.weekly_hours = weekly_hours
        self.duration = duration
        self.start_time = start_time
    
    # Is the worker qualified for this role?
    def worker_qualified(self, worker):
        (this_skill_type, this_level) = self.skill
        for (skill_type, level) in worker.qualifications:
            if skill_type == this_skill_type and level >= this_level:
                return True
        return False
    
    @property    
    def end_time(self):
        return self.start_time + self.duration
    
    # Does the other role overlap this role's start time?
    def overlaps_start_time(self, other_role):
        if other_role.start_time <= self.start_time and other_role.end_time > self.start_time:
            return True
        return False
    
    def __str__(self):
        s = 'RID:' + str(self.rid)
        s += "\nSkill: " + str(self.skill)
        s += "\nWeekly Hours: " + str(self.weekly_hours)
        s += "\nDuration: " + str(self.duration) 
        s += "\nStart Time: " + str(self.start_time) + "\n"
        return s

class Project:
    # @param pid: The ID of this project
    # @param roles: A list of Role objects
    # @param skill_pay_dict: A dict of form {(<skill type>, <level>):hourly_pay}
    # @param revenue_multiple: A number representing the multiple of the the total cost to determine the project revenue
    # @param rejection_penalty_multiple: Same as above, for the rejection penalty
    # @param duration: the project duration in weeks
    # @param start_time: an int specifying the starting week
    def __init__(self, pid, roles, skill_pay_dict, revenue_multiple, rejection_penalty_multiple, duration=None, start_time=None):
        self.pid = pid
        self.skill_pay_dict = skill_pay_dict
        self.roles = roles
        self.revenue_multiple = revenue_multiple
        self.rejection_penalty_multiple = rejection_penalty_multiple
        self.duration = duration
        self.start_time = start_time
        for i in self.roles:
            i.duration = self.duration
            i.start_time = start_time   
    
    @property    
    def end_time(self):
        return self.start_time + self.duration
    
    @property
    def base_cost(self):
        return sum([self.skill_pay_dict[role.skill] * role.weekly_hours * self.duration for role in self.roles])
    
    @property
    def revenue(self):
        return round(self.revenue_multiple * self.base_cost, 2)
    
    @property
    def rejection_penalty(self):
        return round(self.rejection_penalty_multiple * self.base_cost, 2)
        
    def get_role_id_pairs(self):
        return [tuple(x) for x in k_comb([r.rid for r in self.roles], 2)]
        
        

class RoleSkillGenerator:
    # @param role_level_pay: A list of lists - each inner list corresponds to a role family/type and each element in an inner list
    #                        corresponds to the distinct skill levels for that role type. Each element in an inner lists the
    #                        base HOURLY pay at the its level for its role family.
    #                        EXAMPLE: [[5,10], [15,20]] means two skills with two levels each. Skill 1 at level 1 gets $5/hour; at level 2  
    #                                  it gets $10/hour. Skill 2 at level 1 gets $15/hour; at level 2 it gets $20/hour.
    # @param weekly_hour_range: A list or tuple with min/max number of weekly/per-time-unit hours of any role.
    # @param worker_num_skill_histogram: A list of relative weights, corresponding to the number of skills a randomly-generated
    #                                    worker will have. Position 0 = 1 skill, position 1 = 2 skills, etc., and the values
    #                                    are relative probability/likelihood weights for that number of skills.
    # @param role_family_names: An optional list of names for each role family.
    # @param hour_increment: The increment to generate hours by
    def __init__(self, role_levels_pay, weekly_hour_range=[5, 40], worker_num_skill_histogram=[1], role_family_names=None, hour_increment=1):
        role_family_names = list(range(len(role_levels_pay))) if role_family_names == None else role_family_names
        self.weekly_hour_range = weekly_hour_range
        self.worker_num_skill_histogram = worker_num_skill_histogram
        self.hour_increment = hour_increment
        self.skills = {}
        for (i, skill) in enumerate(role_family_names):
            for (j, pay) in enumerate(role_levels_pay[i]):
                self.skills[(skill, j)] = pay
        self.n_generated = 0 # number of actual project roles generated        
    
    # Gets a specified number of worker skills in form of (skill type, level). Never selects the same skill type/family twice.
    def select_n_random_skills(self, n):
        selected = []
        skills = list(self.skills.keys())
        while n > 0 and len(skills) > 0:
            (skill, level) = rnd.sample(skills, 1)[0]
            selected.append((skill, level))
            skills = [(x, y) for (x, y) in skills if x != skill]
            n -= 1
        return selected
    
    # Uses the probability histogram to determine the number of skills to generate for a worker. 
    def random_worker_skills(self):
        n = 1 + random_weighted_index(self.worker_num_skill_histogram)
        return self.select_n_random_skills(n)

    
    # Generates a random Role object. 
    def gen_random_role(self):
        skills = list(self.skills.keys())
        if len(skills) > 0:
            (family, level) = rnd.sample(skills, 1)[0]
            self.n_generated += 1
            return Role('_'.join([str(family), str(level), str(self.n_generated)]), (family, level), 
                        rnd.choice(range(self.weekly_hour_range[0], self.weekly_hour_range[1] + 1, self.hour_increment)))
        return None


class ProblemGenerator:
    def __init__(self, n_workers, n_projects, n_roles_per_project, role_skill_generator, horizon_length, project_duration_range,
                 revenue_multiple_range, rejection_penalty_multiple_range, worker_weekly_hour_range, assignment_multiple_range, 
                 reassignment_multiple_range, underutilization_hourly_cost_range, n_conflicting_projects=0, interproject_conflicting_roles=True,
                 hour_increment=5): 
        self.n_workers = n_workers
        self.n_projects = n_projects
        self.n_roles_per_project = n_roles_per_project
        self.role_skill_generator = role_skill_generator
        self.horizon_length = horizon_length
        self.project_duration_range = project_duration_range
        self.revenue_multiple_range = revenue_multiple_range 
        self.rejection_penalty_multiple_range = rejection_penalty_multiple_range
        self.worker_weekly_hour_range = worker_weekly_hour_range
        self.assignment_multiple_range = assignment_multiple_range
        self.reassignment_multiple_range = reassignment_multiple_range
        self.underutilization_hourly_cost_range = underutilization_hourly_cost_range
        self.n_conflicting_projects = n_conflicting_projects
        self.interproject_conflicting_roles = interproject_conflicting_roles
        self.hour_increment = hour_increment
    
    # Start a new problem.
    def reset(self):
        self.gen_worker_count = 1
        self.gen_project_count = 1
        self.workers = [] # List of Worker objects
        self.projects = [] # List of Project objects
        self.problem = None
        
    # Build a list of n randomly-generated workers.    
    def build_workers(self, n):
        workers = []
        for i in range(n):
            wid = 'W' + str(self.gen_worker_count)
            self.gen_worker_count += 1
            qualifications = self.role_skill_generator.random_worker_skills()
            max_role_cost = max([self.role_skill_generator.skills[x] for x in qualifications])
            hourly_cost = round(max_role_cost * rand_float(*self.assignment_multiple_range), 0)
            minmax_weekly_hours = sorted([rnd.choice(range(self.worker_weekly_hour_range[0], self.worker_weekly_hour_range[1] + 1, self.hour_increment)), 
                                          rnd.choice(range(self.worker_weekly_hour_range[0], self.worker_weekly_hour_range[1] + 1, self.hour_increment))])
            underutilization_hourly_cost = round(max_role_cost * rand_float(*self.underutilization_hourly_cost_range), 0)
            workers.append(Worker(wid, qualifications, hourly_cost, minmax_weekly_hours, underutilization_hourly_cost))
        return workers
    
    # Build a list of n randomly-generated projects.      
    def build_projects(self, n):
        projects = []
        for i in range(n):
            pid = 'P' + str(self.gen_project_count)
            self.gen_project_count += 1
            roles = [self.role_skill_generator.gen_random_role() for x in range(rnd.randint(*self.n_roles_per_project))]
            skill_pay_dict = self.role_skill_generator.skills
            revenue_multiple = rand_float(*self.revenue_multiple_range)
            rejection_penalty_multiple = rand_float(*self.rejection_penalty_multiple_range)
            duration = rnd.randint(*self.project_duration_range)
            start_time = rnd.randint(0, self.horizon_length - duration)
            projects.append(Project(pid, roles, skill_pay_dict, revenue_multiple, rejection_penalty_multiple, 
                                   duration=duration, start_time=start_time))
        return projects
    
    @property
    def roles(self):
        return ft.reduce(lambda x,y: x + y, [p.roles for p in self.projects] + [])
    
    # Builds a list of conflicting roles randomly generated between roles in two project sets/lists.
    # @param project_list_1: The first project list
    # @param project_list_2: The second project list
    # @param n_conflicting_projects: The number of extra conflicts to generate
    def build_conflicting_role_list(self, project_list_1, project_list_2, n_conflicting_projects):
        v = set()
        if self.interproject_conflicting_roles:
            for project in project_list_1 + project_list_2:
                v = v.union(project.get_role_id_pairs())
        all_project_pairs = [tuple(x) for x in cartesian_prod(project_list_1, project_list_2) if x[0].pid != x[1].pid]
        conflicting_projects = rnd.sample(list(set(all_project_pairs).difference(v)), n_conflicting_projects)
        for (p1, p2) in conflicting_projects:
            conflicting_roles = [tuple(x) for x in cartesian_prod([r.rid for r in p1.roles], [r.rid for r in p2.roles])]
            v = v.union(conflicting_roles)
        return list(v)
    
    # This builds a new Problem object.
    def build_new_problem(self):
    
        '''
        # The Problem class elements that need to be set here:
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
        '''
        
        self.reset()
        self.workers = self.build_workers(self.n_workers)
        self.projects = self.build_projects(self.n_projects)
        H = [h.wid for h in self.workers]
        I = [i.rid for i in self.roles]
        J = [j.pid for j in self.projects]
        T = self.horizon_length
        T_role = {r.rid:r.duration for r in self.roles}
        
        b = []
        d = {}
        for role in self.roles:
            row = []
            for project in self.projects:
                if role in project.roles:
                    b.append((role.rid, project.pid))
            d[role.rid] = role.weekly_hours
        
        c_plus = {}
        c_minus = {}
        c_penalty = {}
        q = []
        t_plus = {}
        t_minus = {}
        for h in self.workers:
            c_penalty[h.wid] = h.underutilization_hourly_cost
            t_minus[h.wid] = h.minmax_weekly_hours[0]
            t_plus[h.wid] = h.minmax_weekly_hours[1]
            for i in self.roles:
                if i.worker_qualified(h):
                    assign_mult = rand_float(*self.assignment_multiple_range)
                    reassign_mult = rand_float(*self.reassignment_multiple_range)
                    c_plus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * assign_mult, 2)
                    c_minus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * reassign_mult, 2)
                    q.append((h.wid, i.rid))
            
        r_plus, r_minus = {}, {}
        for j in self.projects:
            r_plus[j.pid] = j.revenue
            r_minus[j.pid] = j.rejection_penalty
        
        u = [(i.rid, k.rid) for i in self.roles for k in self.roles if k.overlaps_start_time(i)]
        v = self.build_conflicting_role_list(self.projects, self.projects, self.n_conflicting_projects)
        w = []
        p = Problem(H, I, J, T, T_role, b, c_plus, c_minus, c_penalty, d,
                    q, r_plus, r_minus, t_plus, t_minus, u, v, w)
        self.problem = p
    
    # Add the specified number of randomly-generated new workers and projects. 
    # Also generate the specified number of conflicting project pairs (in addition to any inter-project conflicts).
    def add(self, n_new_workers, n_new_projects, n_new_conflicting_project_pairs=0):
        '''
        # The Problem class elements that need to be set here:
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
        '''
        if self.problem != None:
            new_workers = self.build_workers(n_new_workers)
            new_projects = self.build_projects(n_new_projects)
            new_roles = [r for p in new_projects for r in p.roles]
            new_H = [h.wid for h in new_workers]
            new_I = [i.rid for i in new_roles]
            new_J = [j.pid for j in new_projects]
            new_T_role = {i.rid:i.duration for i in new_roles}
            new_b = [(i.rid, j.pid) for j in new_projects for i in j.roles]
            new_d = {i.rid:i.weekly_hours for i in new_roles}
            new_c_plus, new_c_minus, new_c_penalty, new_q = {}, {}, {}, []
            for h in new_workers:
                new_c_penalty[h.wid] = h.underutilization_hourly_cost
                for i in self.roles + new_roles:
                    if i.worker_qualified(h):
                        assign_mult = rand_float(*self.assignment_multiple_range)
                        reassign_mult = rand_float(*self.reassignment_multiple_range)
                        new_c_plus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * assign_mult, 2)
                        new_c_minus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * reassign_mult, 2)
                        new_q.append((h.wid, i.rid))
                        
            for h in self.workers:
                for i in new_roles:
                    if i.worker_qualified(h):
                        assign_mult = rand_float(*self.assignment_multiple_range)
                        reassign_mult = rand_float(*self.reassignment_multiple_range)
                        new_c_plus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * assign_mult, 2)
                        new_c_minus[(h.wid, i.rid)] = round(h.hourly_cost * i.duration * i.weekly_hours * reassign_mult, 2)
                        new_q.append((h.wid, i.rid))
            
            new_r_minus = {j.pid:j.rejection_penalty for j in new_projects}
            new_r_plus = {j.pid:j.revenue for j in new_projects}
            new_t_minus = {h.wid:h.minmax_weekly_hours[0] for h in new_workers}
            new_t_plus = {h.wid:h.minmax_weekly_hours[1] for h in new_workers}
            new_u = [(i.rid, k.rid) for i in self.roles + new_roles for k in self.roles + new_roles if k.overlaps_start_time(i)]
            new_v = self.build_conflicting_role_list(self.projects + new_projects, new_projects, n_new_conflicting_project_pairs)
            
            self.problem.add_components(H=new_H, I=new_I, J=new_J, T_role=new_T_role, b=new_b, c_plus=new_c_plus, c_minus=new_c_minus, 
                                        c_penalty=new_c_penalty, d=new_d, q=new_q, r_plus=new_r_plus, r_minus=new_r_minus, 
                                        t_plus=new_t_plus, t_minus=new_t_minus, u=new_u, v=new_v)                                        
            self.workers += new_workers
            self.projects += new_projects
            
    # Randomly delete the specified number of workers and projects.        
    def delete(self, n_workers, n_projects):
        if self.problem != None:
            togo_worker_ids = [h.wid for h in rnd.sample(self.workers, n_workers)] # TODO: Wrong! Don't delete, set c_plus[h,i] = M for deleted workers h. Need to correct in core.Problem.delete_workers!
            togo_project_ids = [j.pid for j in rnd.sample(self.projects, n_projects)] # TODO: Wrong! Don't delete, set c_minus[h,i] = M for deleted workers h. Need to correct in core.Problem.delete_projects!
            #self.workers = [h for h in self.workers if not(h.wid in togo_worker_ids)]
            #self.projects = [j for j in self.projects if not (j.pid in togo_project_ids)]
            self.problem.delete_components(H=togo_worker_ids, J=togo_project_ids)
               
            
    # This updates (i.e. overwrites) the self.w model input (previous assignments) with a set of new assignments, (the new_w_input).
    # @param new_w_input: An array[H, I] of int: w;  % = 1 if worker h assigned to role i in previous allocation, 0 otherwise
    # ** IMPORTANT NOTE: when updating, this MUST be called before any component additions/deletions to maintain consistency.
    def update_w_input(self, new_w_input):
        if self.problem != None:
            self.problem.update_w_input(new_w_input)
   