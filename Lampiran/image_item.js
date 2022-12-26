import React from "react"

const ImageItem = ({ data, is_selected, clickFunction }) => {
    const style = {
        minWidth: '12vw'
    }
    return(
        <button className={"btn btn-light float-start p-3 m-1 border rounded text-start " + (is_selected ? "border-primary": "")} style={style} onClick={clickFunction}>
            <div className="h5">
                {data.image_name}
            </div>
            <div className="fw-light h5">
                {data.total_weight}
            </div>
        </button>
    )
}

export default ImageItem