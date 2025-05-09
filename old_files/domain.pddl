(define (domain pick_and_place)
  (:predicates
    (gripper_empty)
    (on_table ?block)
    (clear ?block)
    (holding ?block)
    (on ?block ?target)  ;; Added on predicate
  )

  ;; Operator to pick up a block
  (:action pick
    :parameters (?block)
    :precondition (and (gripper_empty) (clear ?block) (on_table ?block))
    :effect (and (not (gripper_empty)) (holding ?block) (not (on_table ?block)) (not (clear ?block)))
  )

  ;; Operator to place a block on another block
  (:action place
    :parameters (?block ?target)
    :precondition (and (holding ?block) (clear ?target))
    :effect (and (not (holding ?block)) (gripper_empty) (not (clear ?target)) (clear ?block) (on ?block ?target))
  )

  ;; Operator to place a block on the table
  (:action putontable
    :parameters (?block)
    :precondition (holding ?block)
    :effect (and (not (holding ?block)) (gripper_empty) (on_table ?block) (clear ?block))
  )
)
