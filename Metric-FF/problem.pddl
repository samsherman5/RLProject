(define (problem pick_and_place_problem)
  (:domain pick_and_place)

  ;; Define the objects (blocks)
  (:objects
    blue red green yellow
  )

  ;; Define the initial state
  (:init
    (gripper_empty)
    (on_table blue)
    (on_table red)
    (on_table green)
    (on_table yellow)
    (clear blue)
    (clear red)
    (clear green)
    (clear yellow)
  )

  ;; Define the goal state: All blocks are stacked
  (:goal
    (and
      (on_table blue)   ;; blue is on table
      (on red blue)  ;; red is on blue
      (on green red)  ;; green is on red
      (clear green)      ;; green is the top and clear
      (on_table yellow)  ;; yellow is on table
      (clear yellow)  ;; yellow is clear
    )
  )
)
