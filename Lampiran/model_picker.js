import React, { useState, useEffect } from "react"
import axios from "axios"
import RangeSlider from "./range_slider"
import url from '../config/urls'

const ModelPicker = ({ changeUniqueness, changeConsistency, changeModel }) => {
    const [models, setModels] = useState([])

    const _retrieveModels = () => {
        axios.get(url.app.models)
            .then((res) => {
                setModels(res.data)
                changeModel(res.data[0])
            })
    }

    useEffect(() => {
        _retrieveModels()
    }, [])

    const pickModel = (event) => {
        changeModel(event.target.value)
    }

    return(
        <div className="container">
            <div className="row">
                <select className="form-select" onChange={pickModel}>
                    {
                        models.map((m) => {
                            return <option value={m} key={m}>{m}</option>
                        })
                    }
                </select>
            </div>
            <div className="row">
                <div>
                    <span className="fw-lighter">Min. Uniqueness</span>
                    <RangeSlider changeFunction={changeUniqueness}/>
                </div>
                <div>
                    <span className="fw-lighter">Min. Consistency</span>
                    <RangeSlider changeFunction={changeConsistency}/>
                </div>
            </div>
        </div>
    )
}

export default ModelPicker
