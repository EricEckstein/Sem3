using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DisableDistance : MonoBehaviour
{
    Camera m_MainCamera;
    GameObject[] turtles;

    void Start()
    {
        m_MainCamera = Camera.main;
        turtles = GameObject.FindGameObjectsWithTag("Turtle");
    }

    // Update is called once per frame
    void Update()
    {
        foreach(var turtle in turtles)
        {

            if(Vector3.Distance(turtle.transform.position, m_MainCamera.transform.position) > 15)
            {
                turtle.SetActive(false);
            }
            else
            {
                turtle.SetActive(true);
            }

            if (Vector3.Distance(turtle.transform.position, m_MainCamera.transform.position) < 3)
            {
                var animator = turtle.GetComponent<Animator>();
                animator.Play("Jump");
            }
        }
    }
}
