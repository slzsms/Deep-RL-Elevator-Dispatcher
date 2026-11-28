# '''
# Active hall calls are assigned to the closest available elevator.
# Available elevators are empty and have closed doors.
# Each elevator receives at most one hall call per assignment cycle.
# Assignments are recalculated at every time step based on current
# elevator positions and active hall calls.
# '''


class NearestCarAgent:
    def __init__(self, num_elevators):
        '''
        Initialize the agent with the following attributes:
        - num_elevators: number of elevators in the environment
        '''
        
        self.num_elevators = num_elevators
 
    def act(self, observation):
        '''
        Given an observation:
        - identify active hall calls (floors with waiting passengers)
        - identify available elevators
        - assign the closest elevator-call combinations (excluding calls that are currently being serviced)
        - generate and return the action list
            action list = list of integers length 2*num_elevators where each pair of numbers represents an elevator 
            [dispatch_target, elevator_direction]
            - dispatch_target: the floor to which the elevator is dispatched (between 1 and maxfloornumber=10)
            - elevator_direction: the direction in which the elevator should move (-1=down, 0=no direction, 1=up)
        '''
        
        # get active calls from the observation
        active_calls = get_active_calls(observation)
        
        # update assignments list to remove serviced calls and add new calls
        assignments = assign_nearest_elevator(observation, active_calls, self.num_elevators)

        # generate action
        action = generate_action(observation, assignments, self.num_elevators)
        
        # PRINT STATEMENTS FOR DEBUGGING   
        # print(f"Active Calls: {active_calls}")
        # print(f"Elevator Assignments: {assignments}")
        # print(f"Action List: {action}") 

        # for elevatorState in observation.ElevatorStates:
        #     print(
        #         f"Floor: {elevatorState.Floor}, "
        #         f"Maximum Floor: {elevatorState.MaximumFloor}, "
        #         f"Velocity: {elevatorState.Velocity}, "
        #         f"Maximum Speed: {elevatorState.MaximumSpeed}, "
        #         f"Direction: {elevatorState.Direction}, "
        #         f"Door State: {elevatorState.DoorState}, "
        #         f"Current Dispatch Target: {elevatorState.CurrentDispatchTarget}, "
        #         f"Dispatch Target Direction: {elevatorState.DispatchTargetDirection}, "
        #         f"Load Weight: {elevatorState.LoadWeight}, "
        #         f"Maximum Load: {elevatorState.MaximumLoad}, "
        #         f"Reserved Target Floors: {elevatorState.ReservedTargetFloors}, "
        #         f"Overloaded Alarm: {elevatorState.OverloadedAlarm}, "
        #         f"Door Is Opening: {elevatorState.DoorIsOpening}, "
        #         f"Door Is Closing: {elevatorState.DoorIsClosing}"
        #     )          
        return action
    
   
### HELPER FUNCTIONS 
def get_active_calls(observation):
    '''
    Given an observation, return a set of active calls: (floor, direction)
    '''
    
    # create an empty list to store active calls
    active_calls = []

    # add upward floors with direction = 1
    for floor in observation.RequiringUpwardFloors:
        call = (floor, 1)
        
        # check if it exists before adding for uniqueness
        if call not in active_calls:
            active_calls.append(call)

    # add downward floors with direction =-1
    for floor in observation.RequiringDownwardFloors:
        call = (floor, -1)
        
        # check if it exists before adding for uniqueness
        if call not in active_calls:
            active_calls.append(call)

    return active_calls


def get_available_elevators(observation, num_elevators):
    """
    Return elevators that are empty with doors closed
    """

    available_elevators = []

    for elevator in range(num_elevators):
        elevator_state = observation.ElevatorStates[elevator]

        # elevator is available if there are no passengers are doors are closed
        # ensures that an elevator carrying passengers is not considered available for a new hall cal
        # don't want an elevator that has just arrived at a pickup floor to be immediately assigned another call while its doors are operating.
        if elevator_state.LoadWeight < 1 and elevator_state.DoorState == 0:
            available_elevators.append(elevator)

    return available_elevators

def calculate_distance(elevator_floor, call_floor):
    """
    Calculate the distance between an elevator and a hall call.
    """
    distance = abs(elevator_floor - call_floor)
    
    return distance

def get_unassigned_calls(observation, active_calls):
    """
    Exclude calls already being serviced by an elevator (elevator already at that floor and doors opening)
    """

    unassigned_calls = []

    for call in active_calls:

        call_floor, call_direction = call
        already_being_serviced = False

        for elevator_state in observation.ElevatorStates:

            # if elevator already at floor and doors open then dont include this call to be reassigned to another elevator
            if (elevator_state.Floor == call_floor and elevator_state.DoorState > 0):
                already_being_serviced = True
                break

        if not already_being_serviced:
            unassigned_calls.append(call)

    return unassigned_calls

def assign_nearest_elevator(observation, active_calls, num_elevators):
    """
    Assign active hall calls to the closest available elevators.
    """

    assignments = {}

    available_elevators = get_available_elevators(observation, num_elevators)

    # store active calls that have not been assigned
    unassigned_calls = get_unassigned_calls(observation, active_calls)

    # continue until no available elevators or calls remain
    while available_elevators and unassigned_calls:

        # initialize closest elevator-call combination
        min_distance = float("inf")
        nearest_elevator = None
        nearest_call = None

        # evaluate every available elevator against every active call
        for elevator in available_elevators:

            elevator_floor = observation.ElevatorStates[elevator].Floor

            for call in unassigned_calls:

                call_floor, call_direction = call

                distance = calculate_distance(elevator_floor, call_floor)

                # update closest elevator-call combination
                if distance < min_distance:
                    min_distance = distance
                    nearest_elevator = elevator
                    nearest_call = call

        # assign closest elevator to closest call
        assignments[nearest_call] = nearest_elevator

        # remove assigned elevator and call
        available_elevators.remove(nearest_elevator)
        unassigned_calls.remove(nearest_call)

    return assignments

def generate_action(observation, assignments, num_elevators):
    """
    Convert elevator assignments into a LiftSim action
    """
    # initiate default action list of 0's
    action = [-1, 0] * (num_elevators)

    # set dispatch targets for assigned elevators
    for call, elevator in assignments.items():

        call_floor, call_direction = call
        
        elevator_state = observation.ElevatorStates[elevator]

        # do not keep sending action if elevator already at the pickup floor (it will repeatedly reset the elevator open/door close and get stuck)
        if elevator_state.Floor == call_floor and elevator_state.DoorState > 0:
            continue

        action[2 * elevator] = call_floor
        action[2 * elevator + 1] = call_direction

    return action
