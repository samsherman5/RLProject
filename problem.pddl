(define (problem pick_and_place_problem)
  (:domain pick_and_place)

  ;; Define the objects (blocks)
  (:objects
    blue1 blue2 green1 green2 yellow1 yellow2 purple1 purple2 red
  )

  ;; Define the initial state
  (:init
    (gripper_empty)
    (on_table blue1)
    (on_table blue2)
    (on_table yellow1)
    (on_table yellow2)
    (on_table green1)
    (on_table green2)
    (on_table red)
    (on_table purple1)
    (on_table purple2)

    (clear red)
    (clear blue1)
    (clear blue2)
    (clear yellow1)
    (clear yellow2)
    (clear green1)
    (clear green2)
    (clear purple1)
    (clear purple2)
  )

  ;; Define the goal state: All blocks are stacked
  (:goal
    (and
      (on_table blue1)   ;; blue1 is on table
      (on_table blue2)   ;; blue2 is on table
      (on_table red)   ;; red is on table
      (on purple1 blue1)  ;; purple1 is on blue1
      (on purple2 blue2)  ;; purple2 is on blue2
      (on yellow2 purple1)  ;; yellow2 is on purple1
      (on yellow1 purple2)  ;; yellow1 is on purple2
      (clear yellow1)      ;; yellow1 is the top and clear
      (clear yellow2)      ;; yellow2 is the top and clear
    )
  )
)
